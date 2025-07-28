#include "LowerToCalyxUtil.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxLoweringUtils.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "convertPattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cassert>

using namespace circt;
using namespace circt::lowertocalyx;
using namespace mlir;

namespace circt {
namespace lowertocalyx {

LogicalResult FuncFuncToCalyxPattern::matchAndRewrite(
    mlir::func::FuncOp op, mlir::func::FuncOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Check if the function has only one return statement
  SmallVector<mlir::func::ReturnOp> returnOps;
  op.walk(
      [&](mlir::func::ReturnOp returnOp) { returnOps.push_back(returnOp); });

  if (returnOps.size() != 1) {
    return rewriter.notifyMatchFailure(
        op, "Function must have exactly one return statement");
  }

  auto loc = op.getLoc();
  auto funcType = op.getFunctionType();
  auto inputTypes = funcType.getInputs();
  auto outputTypes = funcType.getResults();

  // Create port info for the component
  SmallVector<calyx::PortInfo> ports;

  // Add input ports
  for (size_t i = 0; i < inputTypes.size(); ++i) {
    ports.push_back({rewriter.getStringAttr("arg" + std::to_string(i)),
                     inputTypes[i], calyx::Direction::Input,
                     DictionaryAttr::get(rewriter.getContext())});
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

  // Move the scaffolded structure from function to component
  // Only move the actual wires and control ops, not duplicate them
  auto &funcBlock = op.getBody().front();

  // Create IRMapping to handle value remapping from function args to component
  // args
  mlir::IRMapping mapper;
  auto funcArgs = op.getArguments();
  for (size_t i = 0; i < funcArgs.size(); ++i) {
    Value componentInput = componentOp.getArguments()[i];
    mapper.map(funcArgs[i], componentInput);
  }

  // Update function arguments to component arguments before moving operations
  for (size_t i = 0; i < funcArgs.size(); ++i) {
    Value componentInput = componentOp.getArguments()[i];
    funcArgs[i].replaceAllUsesWith(componentInput);
  }

  // Get the automatically created wires and control operations in the component
  auto componentWiresOp =
      *componentOp.getBodyBlock()->getOps<calyx::WiresOp>().begin();
  auto componentControlOp =
      *componentOp.getBodyBlock()->getOps<calyx::ControlOp>().begin();

  // Find the scaffolded wires and control operations in the function
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

  // Move other operations (registers, constants) to component body BEFORE wires
  // and control Insert them right after the component creation but before
  // wires/control
  for (Operation *op : otherOpsToMove) {
    op->moveBefore(componentWiresOp);
  }

  // Move content from function wires to component wires
  if (funcWiresOp && !funcWiresOp.getBodyRegion().empty()) {
    auto &funcWiresBlock = funcWiresOp.getBodyRegion().front();
    auto &componentWiresBlock = componentWiresOp.getBodyRegion().front();
    componentWiresBlock.getOperations().splice(componentWiresBlock.end(),
                                               funcWiresBlock.getOperations());
  }

  // Move content from function control to component control
  if (funcControlOp && !funcControlOp.getBodyRegion().empty()) {
    auto &funcControlBlock = funcControlOp.getBodyRegion().front();
    auto &componentControlBlock = componentControlOp.getBodyRegion().front();
    componentControlBlock.getOperations().splice(
        componentControlBlock.end(), funcControlBlock.getOperations());
  }

  // Add necessary wires section assignments for component interface
  if (!outputTypes.empty()) {
    // Create done signal at component level, before wires operation
    rewriter.setInsertionPoint(componentWiresOp);

    // Add assignments to wires section
    auto &wiresBlock = componentWiresOp.getBodyRegion().front();
    rewriter.setInsertionPointToStart(&wiresBlock);
    OpBuilder wiresBuilder(&wiresBlock, wiresBlock.end());

    // Connect output port based on the return operand
    // Find the return operation and check what defines its operand
    auto returnOp =
        returnOps[0]; // We already validated there's exactly one return
    if (returnOp.getNumOperands() > 0) {
      Value returnValue = returnOp.getOperand(0);

      // Connect component output to the return value
      Value outputPort = calyx::getComponentOutput(componentOp, 0);
      wiresBuilder.create<calyx::AssignOp>(loc, outputPort, returnValue);

      // Use utility function to resolve the appropriate done signal with constant deduplication
      Value doneSignal = resolveDoneSignalForValue(returnValue, loc, wiresBuilder, componentOp.getOperation());
      
      // Use utility function to update the component's done connection
      updateComponentDoneConnection(componentOp, doneSignal, loc, wiresBuilder);
    }
  }

  // Replace the function operation with the component
  rewriter.replaceOp(op, componentOp);

  return success();
}

LogicalResult FuncCallToCalyxPattern::matchAndRewrite(
    mlir::func::CallOp op, mlir::func::CallOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  return success();
}

LogicalResult FuncReturnToCalyxPattern::matchAndRewrite(
    mlir::func::ReturnOp op, mlir::func::ReturnOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // func.return should just be erased - the output connections
  // should have been handled by the block-to-group phase
  rewriter.eraseOp(op);
  return success();
}

} // namespace lowertocalyx
} // namespace circt