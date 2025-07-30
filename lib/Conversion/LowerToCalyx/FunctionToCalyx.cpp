#include "LowerToCalyxUtil.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxLoweringUtils.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "convertPattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cassert>

using namespace circt;
using namespace circt::lowertocalyx;
using namespace mlir;

namespace circt {
namespace lowertocalyx {

LogicalResult CompleteFuncToComponentPattern::matchAndRewrite(
    mlir::func::FuncOp op, mlir::func::FuncOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Check if function is external (skip these)
  if (op.isExternal()) {
    return failure();
  }
  
  // Check if the function has exactly one return statement
  SmallVector<mlir::func::ReturnOp> returnOps;
  op.walk([&](mlir::func::ReturnOp returnOp) { returnOps.push_back(returnOp); });

  if (returnOps.size() != 1) {
    return rewriter.notifyMatchFailure(
        op, "Function must have exactly one return statement");
  }

  auto loc = op.getLoc();
  auto funcType = op.getFunctionType();
  
  // Use TypeConverter to get converted signature
  const TypeConverter *typeConverter = getTypeConverter();
  SmallVector<Type> inputTypes;
  SmallVector<Type> outputTypes;
  SmallVector<std::pair<size_t, MemRefType>> memrefArgs; // argIndex, memrefType
  
  // Process input types using TypeConverter
  for (size_t i = 0; i < funcType.getInputs().size(); ++i) {
    Type inputType = funcType.getInputs()[i];
    if (auto memrefType = dyn_cast<MemRefType>(inputType)) {
      // memref arguments become internal memory - skip in component signature
      memrefArgs.push_back({i, memrefType});
    } else {
      // Use TypeConverter for other types
      Type convertedType = typeConverter->convertType(inputType);
      if (convertedType) {
        inputTypes.push_back(convertedType);
      }
    }
  }
  
  // Process output types using TypeConverter
  for (Type outputType : funcType.getResults()) {
    Type convertedType = typeConverter->convertType(outputType);
    if (convertedType) {
      outputTypes.push_back(convertedType);
    }
  }

  // Create component ports - keep it simple
  SmallVector<calyx::PortInfo> ports;

  // Add input ports (non-memref arguments only)
  size_t portIndex = 0;
  for (size_t i = 0; i < funcType.getInputs().size(); ++i) {
    // Skip memref arguments
    bool isMemref = false;
    for (auto &memrefArg : memrefArgs) {
      if (memrefArg.first == i) {
        isMemref = true;
        break;
      }
    }
    if (!isMemref) {
      ports.push_back({rewriter.getStringAttr("arg" + std::to_string(portIndex)),
                       inputTypes[portIndex], calyx::Direction::Input,
                       DictionaryAttr::get(rewriter.getContext())});
      portIndex++;
    }
  }

  // Add output ports
  for (size_t i = 0; i < outputTypes.size(); ++i) {
    ports.push_back({rewriter.getStringAttr("out" + std::to_string(i)),
                     outputTypes[i], calyx::Direction::Output,
                     DictionaryAttr::get(rewriter.getContext())});
  }

  // Add mandatory component ports (clk, reset, go, done)
  calyx::addMandatoryComponentPorts(rewriter, ports);

  // Create the component operation
  auto componentOp = rewriter.create<calyx::ComponentOp>(
      loc, rewriter.getStringAttr(op.getName()), ports);

  // Create internal memory operations for memref arguments
  SmallVector<calyx::SeqMemoryOp> memoryOps;
  rewriter.setInsertionPoint(componentOp.getWiresOp());
  
  for (auto &memrefArg : memrefArgs) {
    size_t argIndex = memrefArg.first;
    MemRefType memrefType = memrefArg.second;
    
    if (!memrefType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "Dynamic memref shapes not supported");
    }
    
    // Calculate memory parameters
    auto shape = memrefType.getShape();
    int64_t elementWidth = memrefType.getElementTypeBitWidth();
    int64_t size = 1;
    for (auto dim : shape) {
      size *= dim;
    }
    
    // Create memory name
    std::string memName = "mem_arg_" + std::to_string(argIndex);
    
    // Create the memory operation
    auto memOp = rewriter.create<calyx::SeqMemoryOp>(
        loc, memName, elementWidth, size, /*readLatency=*/1);
    memoryOps.push_back(memOp);
  }

  // Create value mapping between function arguments and component arguments
  mlir::IRMapping valueMapping;
  auto funcArgs = op.getArguments();
  auto componentArgs = componentOp.getArguments();
  
  // Map function arguments to component arguments and memory operations
  // Component arguments order: input ports (non-memref), output ports, clk, reset, go, done
  size_t componentArgIndex = 0;
  size_t memoryIndex = 0;
  
  for (size_t i = 0; i < funcArgs.size(); ++i) {
    // Check if this argument is a memref
    bool isMemref = false;
    for (auto &memrefArg : memrefArgs) {
      if (memrefArg.first == i) {
        isMemref = true;
        break;
      }
    }
    
    if (isMemref) {
      // For memref arguments, don't create mappings since they'll be handled by memory patterns
      // The memory patterns will find these by argument index after function conversion
      memoryIndex++;
    } else {
      // Map regular arguments to component ports
      if (componentArgIndex < componentArgs.size()) {
        valueMapping.map(funcArgs[i], componentArgs[componentArgIndex]);
        componentArgIndex++;
      }
    }
  }

  // Find the scaffolded wires and control operations in the function
  auto &funcBlock = op.getBody().front();
  calyx::WiresOp funcWiresOp = nullptr;
  calyx::ControlOp funcControlOp = nullptr;
  SmallVector<Operation *> otherOpsToMove;

  for (Operation &bodyOp : funcBlock.getOperations()) {
    if (auto wiresOp = dyn_cast<calyx::WiresOp>(bodyOp)) {
      funcWiresOp = wiresOp;
    } else if (auto controlOp = dyn_cast<calyx::ControlOp>(bodyOp)) {
      funcControlOp = controlOp;
    } else if (!isa<mlir::func::ReturnOp>(bodyOp)) {
      otherOpsToMove.push_back(&bodyOp);
    }
  }

  // Get the component's wires and control operations
  auto componentWiresOp = componentOp.getWiresOp();
  auto componentControlOp = componentOp.getControlOp();

  // Clone other operations (registers, constants) to component body with value mapping
  rewriter.setInsertionPoint(componentWiresOp);
  for (Operation *op : otherOpsToMove) {
    rewriter.clone(*op, valueMapping);
  }

  // Clone content from function wires to component wires with value mapping
  if (funcWiresOp && !funcWiresOp.getBodyRegion().empty()) {
    auto &funcWiresBlock = funcWiresOp.getBodyRegion().front();
    auto &componentWiresBlock = componentWiresOp.getBodyRegion().front();
    
    rewriter.setInsertionPointToEnd(&componentWiresBlock);
    for (auto &op : llvm::make_early_inc_range(funcWiresBlock)) {
      rewriter.clone(op, valueMapping);
    }
  }

  // Clone content from function control to component control with value mapping
  if (funcControlOp && !funcControlOp.getBodyRegion().empty()) {
    auto &funcControlBlock = funcControlOp.getBodyRegion().front();
    auto &componentControlBlock = componentControlOp.getBodyRegion().front();
    
    rewriter.setInsertionPointToEnd(&componentControlBlock);
    for (auto &op : llvm::make_early_inc_range(funcControlBlock)) {
      if (!isa<mlir::func::ReturnOp>(op)) {
        rewriter.clone(op, valueMapping);
      }
    }
  }

  // Handle return value connection
  if (!outputTypes.empty()) {
    auto &wiresBlock = componentWiresOp.getBodyRegion().front();
    rewriter.setInsertionPointToStart(&wiresBlock);
    OpBuilder wiresBuilder(&wiresBlock, wiresBlock.end());

    // Find the return operation and connect its operand to component output
    auto returnOp = returnOps[0];
    if (returnOp.getNumOperands() > 0) {
      Value originalReturnValue = returnOp.getOperand(0);
      
      // Look up the mapped return value if it exists in our mapping
      Value returnValue = valueMapping.lookupOrDefault(originalReturnValue);
      
      // If the return value wasn't mapped and comes from a memref operation,
      // defer the output connection to be handled by memory patterns
      if (returnValue == originalReturnValue) {
        // Check if the return value comes from a memref operation
        if (auto definingOp = originalReturnValue.getDefiningOp()) {
          if (isa<mlir::memref::LoadOp>(definingOp)) {
            // Skip output connection for now - memory patterns will handle this
          } else {
            // Connect component output to the return value
            Value outputPort = calyx::getComponentOutput(componentOp, 0);
            wiresBuilder.create<calyx::AssignOp>(loc, outputPort, returnValue);

            // Set up done signal
            Value doneSignal = resolveDoneSignalForValue(
                returnValue, loc, wiresBuilder, componentOp.getOperation());
            updateComponentDoneConnection(componentOp, doneSignal, loc, wiresBuilder);
          }
        }
      } else {
        // Connect component output to the return value
        Value outputPort = calyx::getComponentOutput(componentOp, 0);
        wiresBuilder.create<calyx::AssignOp>(loc, outputPort, returnValue);

        // Set up done signal
        Value doneSignal = resolveDoneSignalForValue(
            returnValue, loc, wiresBuilder, componentOp.getOperation());
        updateComponentDoneConnection(componentOp, doneSignal, loc, wiresBuilder);
      }
    }
  }

  // Replace the function operation with the component
  rewriter.replaceOp(op, componentOp);

  return success();
}

LogicalResult FuncReturnToCalyxPattern::matchAndRewrite(
    mlir::func::ReturnOp op, mlir::func::ReturnOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // func.return should just be erased - the output connections
  // should have been handled by the component conversion
  rewriter.eraseOp(op);
  return success();
}

} // namespace lowertocalyx
} // namespace circt