#include "LowerToCalyxUtil.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxLoweringUtils.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "convertPattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

#include <cassert>

using namespace circt;
using namespace circt::lowertocalyx;
using namespace mlir;

namespace circt {
namespace lowertocalyx {

static void storeLowering(mlir::ConversionPatternRewriter &rewriter,
                          mlir::memref::StoreOp &storeOp,
                          circt::calyx::SeqMemoryOp &memOp,
                          const mlir::TypeConverter *typeConverter,
                          const mlir::Location &loc,
                          const circt::calyx::WiresOp &wiresOp,
                          mlir::func::FuncOp &op) {
  // Use only the ConversionPatternRewriter to avoid insertion point conflicts
  rewriter.setInsertionPointAfter(storeOp);

  Value addrPort = memOp.addrPort(0);
  Value indexValue = storeOp.getIndices()[0];

  // Simplified approach: just assign index value directly to address port
  // The conversion framework will handle type conversion later in the pipeline
  rewriter.create<calyx::AssignOp>(loc, addrPort, indexValue);
  rewriter.create<calyx::AssignOp>(loc, memOp.writeData(),
                                   storeOp.getValueToStore());

  // Create constant 1 for contentEn
  auto constantOp = rewriter.create<hw::ConstantOp>(
      loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
  rewriter.create<calyx::AssignOp>(loc, memOp.contentEn(),
                                   constantOp.getResult());

  rewriter.eraseOp(storeOp);
}

static void loadLowering(mlir::ConversionPatternRewriter &rewriter,
                         mlir::memref::LoadOp &loadOp,
                         circt::calyx::SeqMemoryOp &memOp,
                         const mlir::TypeConverter *typeConverter,
                         const mlir::Location &loc,
                         const circt::calyx::WiresOp &wiresOp,
                         mlir::func::FuncOp &op) {
  // Use only the ConversionPatternRewriter to avoid insertion point conflicts
  rewriter.setInsertionPointAfter(loadOp);

  Value addrPort = memOp.addrPort(0);
  Value indexValue = loadOp.getIndices()[0];

  // Simplified approach: just assign index value directly to address port
  // The conversion framework will handle type conversion later in the pipeline
  rewriter.create<calyx::AssignOp>(loc, addrPort, indexValue);

  // Create constant 1 for contentEn
  auto constantOp = rewriter.create<hw::ConstantOp>(
      loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
  rewriter.create<calyx::AssignOp>(loc, memOp.contentEn(),
                                   constantOp.getResult());

  rewriter.replaceOp(loadOp, memOp.readData());
}

LogicalResult FuncFuncToCalyxPattern::matchAndRewrite(
    mlir::func::FuncOp op, mlir::func::FuncOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Check if function is external (skip these)
  if (op.isExternal()) {
    return failure();
  }

  // Check if the function has exactly one return statement
  SmallVector<mlir::func::ReturnOp> returnOps;
  op.walk(
      [&](mlir::func::ReturnOp returnOp) { returnOps.push_back(returnOp); });

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

  // SIMPLE SCAFFOLDING: Create scaffolding directly without complex logic
  // Simply create wires and control ops if they don't exist
  calyx::WiresOp wiresOp;
  calyx::ControlOp controlOp;

  auto wiresOps = op.getFunctionBody().getOps<calyx::WiresOp>();
  if (wiresOps.empty()) {
    // Create basic scaffolding without complex block manipulation
    rewriter.setInsertionPointToStart(&op.getBody().front());
    wiresOp = rewriter.create<calyx::WiresOp>(loc);
    controlOp = rewriter.create<calyx::ControlOp>(loc);

    // Move all original operations to the control block
    Block *controlBlock = &controlOp.getBodyRegion().front();
    SmallVector<Operation *> opsToMove;
    for (auto &op : op.getBody().front()) {
      if (!isa<calyx::WiresOp, calyx::ControlOp>(op)) {
        opsToMove.push_back(&op);
      }
    }

    for (auto *opToMove : opsToMove) {
      opToMove->moveBefore(controlBlock, controlBlock->end());
    }
  } else {
    wiresOp = *wiresOps.begin();
    auto controlOps = op.getFunctionBody().getOps<calyx::ControlOp>();
    if (controlOps.empty()) {
      return rewriter.notifyMatchFailure(
          op, "Function has wires but no control operation");
    }
    controlOp = *controlOps.begin();
  }

  // Create builder for memory operations in wires section
  OpBuilder componentBuilder(rewriter.getContext());
  componentBuilder.setInsertionPoint(wiresOp);

  // Process input types using TypeConverter - simplified
  for (size_t i = 0; i < funcType.getInputs().size(); ++i) {
    Type inputType = funcType.getInputs()[i];
    if (auto memrefType = dyn_cast<MemRefType>(inputType)) {
      // memref arguments become internal memory - skip in component signature
      memrefArgs.push_back({i, memrefType});

      if (!memrefType.hasStaticShape()) {
        return rewriter.notifyMatchFailure(
            op, "Dynamic memref shapes not supported");
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
      std::string memName = "mem_arg_" + std::to_string(i);

      // Create the memory operation with proper sizes and address widths
      auto memOp = componentBuilder.create<calyx::SeqMemoryOp>(
          loc, memName, elementWidth, sizes, addrSizes);

      if (op.getArgument(i).getNumUses() == 1) {
        auto use = op.getArgument(i).getUses().begin();
        if (auto loadOp = dyn_cast<mlir::memref::LoadOp>(use->getOwner())) {
          loadLowering(rewriter, loadOp, memOp, typeConverter, loc, wiresOp,
                       op);
        } else if (auto storeOp =
                       dyn_cast<mlir::memref::StoreOp>(use->getOwner())) {
          storeLowering(rewriter, storeOp, memOp, typeConverter, loc, wiresOp,
                        op);
        }
      } else {
        for (auto &use : op.getArgument(i).getUses()) {
          if (auto loadOp = dyn_cast<mlir::memref::LoadOp>(use.getOwner())) {
            loadLowering(rewriter, loadOp, memOp, typeConverter, loc, wiresOp,
                         op);

          } else if (auto storeOp =
                         dyn_cast<mlir::memref::StoreOp>(use.getOwner())) {
            storeLowering(rewriter, storeOp, memOp, typeConverter, loc, wiresOp,
                          op);
          }
        }
      }
      op.getArgument(i).dropAllUses();
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
      ports.push_back(
          {rewriter.getStringAttr("arg" + std::to_string(portIndex)),
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

  // Set insertion point to replace the function
  rewriter.setInsertionPoint(op);

  // Create the component operation
  auto componentOp = rewriter.create<calyx::ComponentOp>(
      loc, rewriter.getStringAttr(op.getName()), ports);

  // Handle argument replacement for non-memref arguments
  Block *funcBlock = &op.getBody().front();
  size_t componentArgIndex = 0;
  for (size_t i = 0; i < op.getNumArguments(); ++i) {
    bool isMemref = false;
    for (auto &memrefArg : memrefArgs) {
      if (memrefArg.first == i) {
        isMemref = true;
        break;
      }
    }

    if (!isMemref) {
      Value functionArg = op.getArgument(i);
      Value componentArg = componentOp.getArgument(componentArgIndex);
      componentArgIndex++;

      // Use MLIR's proper API to replace uses within the function region
      replaceAllUsesInRegionWith(functionArg, componentArg, op.getBody());
    }
  }

  // SECOND: Now setup component structure and move operations
  // Get the component's wires block to place operations
  componentOp.getWiresOp().erase();
  componentOp.getControlOp().erase();

  Block *compBlock = componentOp.getBodyBlock();

  // Use rewriter to move operations from function block to component wires
  // block This preserves the operations for other patterns to convert them
  rewriter.setInsertionPointToEnd(compBlock);

  // Move operations one by one using the rewriter
  for (auto &opToMove :
       llvm::make_early_inc_range(funcBlock->getOperations())) {
    opToMove.moveBefore(compBlock, compBlock->end());
  }

  rewriter.eraseOp(op);
  return success();
}

LogicalResult FuncReturnToCalyxPattern::matchAndRewrite(
    mlir::func::ReturnOp op, mlir::func::ReturnOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto componentOp = op->getParentOfType<calyx::ComponentOp>();
  if (!componentOp) {
    return rewriter.notifyMatchFailure(
        op, "Return op not inside a Calyx component");
  }

  auto loc = op.getLoc();

  // Get the wires operation to place assignments
  auto wiresOp = componentOp.getWiresOp();
  if (!wiresOp) {
    return rewriter.notifyMatchFailure(op, "Component has no wires operation");
  }

  Block *componentWiresBlock = wiresOp.getBodyBlock();
  OpBuilder wiresBuilder(componentWiresBlock, componentWiresBlock->end());

  // Map return values to component output ports
  if (!adaptor.getOperands().empty()) {
    for (size_t i = 0; i < adaptor.getOperands().size(); ++i) {
      Value returnValue = adaptor.getOperands()[i];

      // Find the corresponding output port
      // Component ports are: inputs, clk, reset, go, outputs, done
      // We need to find the i-th output port
      auto portInfos = componentOp.getPortInfo();
      size_t outputPortIndex = 0;
      Value outputPort = nullptr;

      for (size_t portIdx = 0; portIdx < portInfos.size(); ++portIdx) {
        if (portInfos[portIdx].direction == calyx::Direction::Output &&
            portInfos[portIdx].name.getValue() != "done") {
          if (outputPortIndex == i) {
            outputPort = componentOp.getArgument(portIdx);
            break;
          }
          outputPortIndex++;
        }
      }

      if (!outputPort) {
        return rewriter.notifyMatchFailure(
            op, "Could not find output port for return value");
      }

      // Create assignment: output_port = return_value
      wiresBuilder.create<calyx::AssignOp>(loc, outputPort, returnValue);
    }
  }

  // Handle done signal based on return value source
  Value doneSignal = nullptr;
  if (!adaptor.getOperands().empty()) {
    Value returnValue = adaptor.getOperands()[0];
    doneSignal = resolveDoneSignalForValue(returnValue, componentOp);
  } else {
    // No return values - use constant 1
    doneSignal = getOrCreateConstant(componentOp, 1);
  }

  // Assign done signal to component done port
  Value componentDonePort = componentOp.getDonePort();
  wiresBuilder.create<calyx::AssignOp>(loc, componentDonePort, doneSignal);

  rewriter.eraseOp(op);
  return success();
}

} // namespace lowertocalyx
} // namespace circt