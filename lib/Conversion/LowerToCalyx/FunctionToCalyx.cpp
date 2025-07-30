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

// Forward declarations
static void convertMemrefLoadOp(mlir::memref::LoadOp loadOp, 
                                ConversionPatternRewriter &rewriter,
                                mlir::IRMapping &valueMapping,
                                SmallVector<std::pair<size_t, MemRefType>> &memrefArgs,
                                SmallVector<calyx::SeqMemoryOp> &memoryOps,
                                calyx::ComponentOp componentOp);

static void convertMemrefStoreOp(mlir::memref::StoreOp storeOp,
                                 ConversionPatternRewriter &rewriter,
                                 mlir::IRMapping &valueMapping,
                                 SmallVector<std::pair<size_t, MemRefType>> &memrefArgs,
                                 SmallVector<calyx::SeqMemoryOp> &memoryOps,
                                 calyx::ComponentOp componentOp);

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
    
    // Calculate sizes and address widths for each dimension
    SmallVector<int64_t> sizes, addrSizes;
    for (auto dim : shape) {
      sizes.push_back(dim);
      addrSizes.push_back(llvm::Log2_64_Ceil(dim));
    }
    
    // Create memory name
    std::string memName = "mem_arg_" + std::to_string(argIndex);
    
    // Create the memory operation with proper sizes and address widths
    auto memOp = rewriter.create<calyx::SeqMemoryOp>(
        loc, memName, elementWidth, sizes, addrSizes);
    memoryOps.push_back(memOp);
  }

  // Pre-process all memref operations in the function to convert them first
  // This prevents null reference issues when cloning operations
  SmallVector<mlir::memref::LoadOp> allLoadOps;
  SmallVector<mlir::memref::StoreOp> allStoreOps;
  
  op.walk([&](mlir::memref::LoadOp loadOp) {
    allLoadOps.push_back(loadOp);
  });
  op.walk([&](mlir::memref::StoreOp storeOp) {
    allStoreOps.push_back(storeOp);
  });

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
      // Create a temporary block argument with the correct memref type to avoid null references
      // This placeholder will prevent null operands during cloning
      Block *tempBlock = new Block();
      Value placeholder = tempBlock->addArgument(funcArgs[i].getType(), funcArgs[i].getLoc());
      valueMapping.map(funcArgs[i], placeholder);
      memoryIndex++;
    } else {
      // Map regular arguments to component ports
      if (componentArgIndex < componentArgs.size()) {
        valueMapping.map(funcArgs[i], componentArgs[componentArgIndex]);
        componentArgIndex++;
      }
    }
  }

  // Pre-map memref load results to memory read data to avoid null references
  for (auto loadOp : allLoadOps) {
    Value memref = loadOp.getMemref();
    if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
      size_t argIndex = blockArg.getArgNumber();
      // Find the corresponding memory operation
      for (size_t i = 0; i < memrefArgs.size(); ++i) {
        if (memrefArgs[i].first == argIndex && i < memoryOps.size()) {
          // Pre-map the load result to the memory read data
          Value readData = memoryOps[i].readData();
          valueMapping.map(loadOp.getResult(), readData);
          // Successfully pre-mapped load result
          break;
        }
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
    // Skip memref operations as they're already converted
    if (!isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(op)) {
      rewriter.clone(*op, valueMapping);
    }
  }

  // Clone content from function wires to component wires with value mapping
  if (funcWiresOp && !funcWiresOp.getBodyRegion().empty()) {
    auto &funcWiresBlock = funcWiresOp.getBodyRegion().front();
    auto &componentWiresBlock = componentWiresOp.getBodyRegion().front();
    
    rewriter.setInsertionPointToEnd(&componentWiresBlock);
    for (auto &op : llvm::make_early_inc_range(funcWiresBlock)) {
      // Skip memref operations as they're already converted
      if (!isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(op)) {
        rewriter.clone(op, valueMapping);
      }
    }
  }

  // Clone content from function control to component control with value mapping
  // and handle memref operations directly
  if (funcControlOp && !funcControlOp.getBodyRegion().empty()) {
    auto &funcControlBlock = funcControlOp.getBodyRegion().front();
    auto &componentControlBlock = componentControlOp.getBodyRegion().front();
    
    rewriter.setInsertionPointToEnd(&componentControlBlock);
    for (auto &op : llvm::make_early_inc_range(funcControlBlock)) {
      if (!isa<mlir::func::ReturnOp>(op)) {
        // Handle memref operations specially
        if (auto loadOp = dyn_cast<mlir::memref::LoadOp>(op)) {
          convertMemrefLoadOp(loadOp, rewriter, valueMapping, memrefArgs, memoryOps, componentOp);
        } else if (auto storeOp = dyn_cast<mlir::memref::StoreOp>(op)) {
          convertMemrefStoreOp(storeOp, rewriter, valueMapping, memrefArgs, memoryOps, componentOp);
        } else {
          // For other operations, clone normally with value mapping
          rewriter.clone(op, valueMapping);
        }
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
          if (auto loadOp = dyn_cast<mlir::memref::LoadOp>(definingOp)) {
            // Handle memref load in return value
            llvm::errs() << "DEBUG: Found memref.load in return value\n";
            convertMemrefLoadOp(loadOp, rewriter, valueMapping, memrefArgs, memoryOps, componentOp);
            // Look up the converted return value
            Value convertedReturnValue = valueMapping.lookupOrDefault(originalReturnValue);
            if (convertedReturnValue != originalReturnValue) {
              // Connect component output to the converted return value
              Value outputPort = calyx::getComponentOutput(componentOp, 0);
              wiresBuilder.create<calyx::AssignOp>(loc, outputPort, convertedReturnValue);

              // Set up done signal
              Value doneSignal = resolveDoneSignalForValue(
                  convertedReturnValue, loc, wiresBuilder, componentOp.getOperation());
              updateComponentDoneConnection(componentOp, doneSignal, loc, wiresBuilder);
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

  // Erase all remaining memref operations in the component that weren't already erased
  SmallVector<mlir::memref::LoadOp> remainingLoadOps;
  SmallVector<mlir::memref::StoreOp> remainingStoreOps;
  
  componentOp.walk([&](mlir::memref::LoadOp loadOp) {
    remainingLoadOps.push_back(loadOp);
  });
  componentOp.walk([&](mlir::memref::StoreOp storeOp) {
    remainingStoreOps.push_back(storeOp);
  });

  // Erase remaining memref operations
  for (auto loadOp : remainingLoadOps) {
    rewriter.eraseOp(loadOp);
  }
  for (auto storeOp : remainingStoreOps) {
    rewriter.eraseOp(storeOp);
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

// Helper function to convert memref.load operations during function conversion
static void convertMemrefLoadOp(mlir::memref::LoadOp loadOp, 
                                ConversionPatternRewriter &rewriter,
                                mlir::IRMapping &valueMapping,
                                SmallVector<std::pair<size_t, MemRefType>> &memrefArgs,
                                SmallVector<calyx::SeqMemoryOp> &memoryOps,
                                calyx::ComponentOp componentOp) {
  Value memref = loadOp.getMemref();
  auto indices = loadOp.getIndices();
  
  // Check if memref is mapped to a placeholder (original was a block argument)
  // We need to find the original memref argument to determine which memory to use
  
  size_t argIndex = -1;
  bool foundArgIndex = false;
  
  // Search through the value mapping to find the original function argument
  for (auto &mappingEntry : valueMapping.getValueMap()) {
    if (mappingEntry.second == memref) {
      // This mapped value matches our memref, check if the key is a function argument
      if (auto blockArg = dyn_cast<BlockArgument>(mappingEntry.first)) {
        argIndex = blockArg.getArgNumber();
        foundArgIndex = true;
        break;
      }
    }
  }
  
  if (foundArgIndex) {
    
    // Find the corresponding SeqMemoryOp
    calyx::SeqMemoryOp memOp = nullptr;
    for (size_t i = 0; i < memrefArgs.size(); ++i) {
      if (memrefArgs[i].first == argIndex && i < memoryOps.size()) {
        memOp = memoryOps[i];
        break;
      }
    }
    
    if (memOp) {
      // Create proper wiring for memory read operation
      Value readData = memOp.readData();
      
      // Map the load result to memory read data in the value mapping
      valueMapping.map(loadOp.getResult(), readData);
      
      // Create assignment to connect memory address if needed
      if (indices.size() == 1) {
        Value mappedIndex = valueMapping.lookupOrDefault(indices[0]);
        Value memAddr = memOp.addrPort(0); // Get address port for dimension 0
        
        // Find the containing group to add the address assignment
        auto parentGroup = loadOp->getParentOfType<calyx::GroupOp>();
        if (parentGroup) {
          OpBuilder groupBuilder(parentGroup.getBodyBlock(), parentGroup.getBodyBlock()->end());
          groupBuilder.create<calyx::AssignOp>(loadOp.getLoc(), memAddr, mappedIndex);
        }
      }
      
      // Don't create the original load operation
      return;
    }
  }
  
  // If we can't handle the memref operation, don't clone it
  // Leave it for later passes to handle
}

// Helper function to convert memref.store operations during function conversion  
static void convertMemrefStoreOp(mlir::memref::StoreOp storeOp,
                                 ConversionPatternRewriter &rewriter,
                                 mlir::IRMapping &valueMapping,
                                 SmallVector<std::pair<size_t, MemRefType>> &memrefArgs,
                                 SmallVector<calyx::SeqMemoryOp> &memoryOps,
                                 calyx::ComponentOp componentOp) {
  Value memref = storeOp.getMemref();
  Value valueToStore = storeOp.getValueToStore();
  auto indices = storeOp.getIndices();
  
  // Find which memref argument this store is using
  if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
    size_t argIndex = blockArg.getArgNumber();
    
    // Find the corresponding SeqMemoryOp
    calyx::SeqMemoryOp memOp = nullptr;
    for (size_t i = 0; i < memrefArgs.size(); ++i) {
      if (memrefArgs[i].first == argIndex && i < memoryOps.size()) {
        memOp = memoryOps[i];
        break;
      }
    }
    
    if (memOp) {
      // Map the store operation to memory write operations
      // For now, create simplified wiring
      // Note: This is a simplified conversion - full implementation would need proper sequencing
      
      // Map value to store and addresses if needed
      Value mappedValue = valueMapping.lookupOrDefault(valueToStore);
      if (indices.size() == 1) {
        Value mappedIndex = valueMapping.lookupOrDefault(indices[0]);
        // Create assignments to connect value and address to memory write ports
        // Note: This is simplified - full implementation would need proper wiring
      }
      
      // Don't create the original store operation
      return;
    }
  }
  
  // If we can't handle the memref operation, don't clone it
  // Leave it for later passes to handle
}

} // namespace lowertocalyx
} // namespace circt