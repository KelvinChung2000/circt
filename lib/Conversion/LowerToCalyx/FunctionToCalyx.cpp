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
                          const mlir::Location &loc,
                          circt::calyx::WiresOp &wiresOp,
                          OpBuilder &wiresBuilder) {

  Value addrPort = memOp.addrPort(0);
  Value indexValue = storeOp.getIndices()[0];
  Value convertedIndexValue = convertValueForToMatchType(
      addrPort, indexValue,
      dyn_cast<calyx::WiresOp>(wiresBuilder.getInsertionBlock()->getParentOp()),
      wiresBuilder, getOpUniqueName(storeOp), loc, rewriter);
  wiresBuilder.create<calyx::AssignOp>(loc, addrPort, convertedIndexValue);
  wiresBuilder.create<calyx::AssignOp>(loc, memOp.writeData(),
                                       storeOp.getValueToStore());
  auto constantOp = getOrCreateConstant(wiresOp, 1, 1);
  wiresBuilder.create<calyx::AssignOp>(loc, memOp.writeEn(), constantOp);
  rewriter.eraseOp(storeOp);
}

static void loadLowering(mlir::ConversionPatternRewriter &rewriter,
                         mlir::memref::LoadOp &loadOp,
                         circt::calyx::SeqMemoryOp &memOp,
                         const mlir::Location &loc,
                         circt::calyx::WiresOp &wiresOp,
                         OpBuilder &wiresBuilder) {

  Value addrPort = memOp.addrPort(0);
  Value indexValue = loadOp.getIndices()[0];
  Value convertedIndexValue = convertValueForToMatchType(
      addrPort, indexValue,
      dyn_cast<calyx::WiresOp>(wiresBuilder.getInsertionBlock()->getParentOp()),
      wiresBuilder, getOpUniqueName(loadOp), loc, rewriter);
  wiresBuilder.create<calyx::AssignOp>(loc, addrPort, convertedIndexValue);
  auto constantOp = getOrCreateConstant(wiresOp, 1, 1);
  wiresBuilder.create<calyx::AssignOp>(loc, memOp.contentEn(), constantOp);
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

  // Collect memref uses for deferred lowering after component creation
  SmallVector<SmallVector<Operation *>> memrefUses; // parallels memrefArgs

  // First pass: classify args, record memref uses only (do not lower yet)
  for (size_t i = 0; i < funcType.getInputs().size(); ++i) {
    Type inputType = funcType.getInputs()[i];
    if (auto memrefType = dyn_cast<MemRefType>(inputType)) {
      // memref arguments become internal memory - skip in component signature
      memrefArgs.push_back({i, memrefType});
      memrefUses.emplace_back();

      if (!memrefType.hasStaticShape()) {
        return rewriter.notifyMatchFailure(
            op, "Dynamic memref shapes not supported");
      }

      for (auto &use : op.getArgument(i).getUses())
        memrefUses.back().push_back(use.getOwner());
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

  // Move remaining ops (loads/stores/return) into component for later passes
  Block *compBlock = componentOp.getControlOp().getBodyBlock();
  for (auto &opToMove : llvm::make_early_inc_range(funcBlock->getOperations()))
    opToMove.moveBefore(compBlock, compBlock->end());

  // Create builder for memory operations in wires section
  auto wiresOp = componentOp.getWiresOp();
  OpBuilder componentBuilder(rewriter.getContext());
  componentBuilder.setInsertionPoint(wiresOp);
  OpBuilder wiresBuilder(wiresOp.getBodyBlock(), wiresOp.getBodyBlock()->end());
  OpBuilder inplaceBuilder(rewriter.getContext());

  for (size_t idx = 0; idx < memrefArgs.size(); ++idx) {
    auto [argIndex, memrefType] = memrefArgs[idx];
    auto shape = memrefType.getShape();
    int64_t elementWidth = memrefType.getElementTypeBitWidth();
    SmallVector<int64_t> sizes, addrSizes;
    for (auto dim : shape) {
      sizes.push_back(dim);
      addrSizes.push_back(llvm::Log2_64_Ceil(dim));
    }
    std::string memName = "mem_arg_" + std::to_string(argIndex);
    auto memOp = componentBuilder.create<calyx::SeqMemoryOp>(
        loc, memName, elementWidth, sizes, addrSizes);
    // Prepare builders: componentBuilder already set before wiresOp; create a
    // dedicated wiresBuilder pointing at end of wires body.
    for (Operation *useOp : memrefUses[idx]) {
      if (auto loadOp = dyn_cast<mlir::memref::LoadOp>(useOp)) {
        inplaceBuilder.setInsertionPointAfter(loadOp);
        loadLowering(rewriter, loadOp, memOp, loc, wiresOp, inplaceBuilder);
      } else if (auto storeOp = dyn_cast<mlir::memref::StoreOp>(useOp)) {
        inplaceBuilder.setInsertionPointAfter(storeOp);
        storeLowering(rewriter, storeOp, memOp, loc, wiresOp, inplaceBuilder);
      }
    }
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