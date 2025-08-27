//===- ControlFlowToCalyx.cpp - Control Flow to Calyx Conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the main functions for converting control flow operations
// to Calyx.
//
//===----------------------------------------------------------------------===//

#include "LowerToCalyxUtil.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "convertPattern.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// Helper function to check if a region contains only side-effect-free
// operations
static bool hasNoSideEffects(Region &region) {
  for (auto &block : region) {
    for (auto &op : block) {
      // Skip yield operations as they're terminators
      if (isa<mlir::scf::YieldOp>(op)) {
        continue;
      }

      // Check if operation has side effects
      if (!isMemoryEffectFree(&op)) {
        return false;
      }

      // Check nested regions recursively
      for (auto &nestedRegion : op.getRegions()) {
        if (!hasNoSideEffects(nestedRegion)) {
          return false;
        }
      }
    }
  }
  return true;
}

/// Transform a single SCF If operation to Calyx hardware constructs
/// This function creates registers, not ops, and assigns directly in the
/// transformation. If both branches have no side effects, it uses MuxLibOp.

// SCF if to Calyx conversion pattern implementation
LogicalResult ScfIfToCalyxPattern::matchAndRewrite(
    mlir::scf::IfOp ifOp, mlir::scf::IfOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Get condition
  Value condition = adaptor.getCondition();

  // Get regions
  auto &thenRegion = ifOp.getThenRegion();
  auto &elseRegion = ifOp.getElseRegion();

  // Get the component containing this operation
  auto componentOp = ifOp->getParentOfType<calyx::ComponentOp>();
  if (!componentOp) {
    return ifOp.emitError("scf.if must be within a calyx.component");
  }

  // Find the wires block
  auto wiresOp = componentOp.getWiresOp();
  if (!wiresOp) {
    return ifOp.emitError("calyx.component must have a wires block");
  }

  // Track the result registers for scf.if
  SmallVector<calyx::RegisterOp> ifResultsRegs;

  // Create registers for each result
  for (size_t i = 0; i < ifOp.getNumResults(); ++i) {
    auto resultType = ifOp.getResult(i).getType();

    // Convert the result type to integer if needed
    if (!resultType.isIntOrIndex()) {
      return ifOp.emitError("scf.if result type must be integer or index");
    }

    unsigned bitWidth = 32; // Default for index type
    if (auto intType = dyn_cast<IntegerType>(resultType)) {
      bitWidth = intType.getWidth();
    }

    // Create register name
    std::string regName =
        getOpUniqueName(ifOp) + "_result_" + std::to_string(i);

    // Set insertion point to before wires block
    rewriter.setInsertionPoint(wiresOp);

    // Create the register
    auto reg =
        rewriter.create<calyx::RegisterOp>(ifOp.getLoc(), regName, bitWidth);

    ifResultsRegs.push_back(reg);
  }

  // Set insertion point inside wires block
  rewriter.setInsertionPointToEnd(wiresOp.getBodyBlock());

  // If there are no results and both regions are simple, convert to a simple
  // Calyx if
  // if (ifOp.getNumResults() == 0 && isSingleAssignment(thenRegion) &&
  //    isSingleAssignment(elseRegion)) {
  //  // Convert to a simple mux operation
  //  auto thenValue = getSingleAssignmentValue(thenRegion);
  //  auto elseValue = getSingleAssignmentValue(elseRegion);
  //  auto dest = getSingleAssignmentDest(thenRegion);

  //  // Create mux name
  //  std::string muxName = "if_mux_" + std::to_string(reinterpret_cast<size_t>(
  //  ifOp.getOperation()));

  //  // Create mux operation
  //  auto muxOp = rewriter.create<calyx::MuxLibOp>(
  //      ifOp.getLoc(), muxName, thenValue.getType());
  //  rewriter.create<calyx::AssignOp>(ifOp.getLoc(), muxOp.getLeft(),
  //  thenValue);
  //  rewriter.create<calyx::AssignOp>(ifOp.getLoc(), muxOp.getRight(),
  //  elseValue);
  //  rewriter.create<calyx::AssignOp>(ifOp.getLoc(), muxOp.getSel(),
  //  condition);
  //  rewriter.create<calyx::AssignOp>(ifOp.getLoc(), dest, muxOp.getOut());

  //   Replace the scf.if with the result of the mux
  //   rewriter.replaceOp(ifOp, muxOp.getOut());
  //   llvm::outs() << "Replaced scf.if with mux: " << muxName << "\n";
  //   return success();
  // }

  // Create constant 1 for write enable using deduplication
  auto constantOne = getOrCreateConstant(componentOp.getOperation(), 1);

  // Replace yield operations with register assignments and group_done
  // operations Then region
  if (!thenRegion.empty()) {
    auto &thenBlock = thenRegion.front();
    for (auto &op : llvm::make_early_inc_range(thenBlock)) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(op)) {
        // Replace std::enumerate with indexed loop
        auto operands = yieldOp.getOperands();
        for (size_t i = 0; i < operands.size(); ++i) {
          auto yieldValue = operands[i];
          // Get the corresponding register for this yield value
          auto &resultReg = ifResultsRegs[i];
          // Create assignment: reg.in = thenYieldValue
          rewriter.setInsertionPoint(&op);
          // Create assignment: reg.in = thenYieldValue
          rewriter.create<calyx::AssignOp>(op.getLoc(), resultReg.getIn(),
                                           yieldValue, ifOp.getCondition());
          // Create assignment: reg.write_en = 1
          rewriter.create<calyx::AssignOp>(op.getLoc(), resultReg.getWriteEn(),
                                           constantOne, ifOp.getCondition());
          // Create group_done
          rewriter.create<calyx::GroupDoneOp>(op.getLoc(), resultReg.getDone());
        }
        // Now safely erase the yield op since we've processed its operands
        rewriter.eraseOp(yieldOp);
      }
    }
  }

  // Else region
  if (!elseRegion.empty()) {
    auto &elseBlock = elseRegion.front();
    for (auto &op : llvm::make_early_inc_range(elseBlock)) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(op)) {
        auto operands = yieldOp.getOperands();
        if (operands.size() == 0) {
          rewriter.eraseOp(yieldOp);
          continue; // Skip the rest of the processing for this yield
        }

        for (size_t i = 0; i < operands.size(); ++i) {
          auto yieldValue = operands[i];
          // Get the corresponding register for this yield value
          auto &resultReg = ifResultsRegs[i];
          rewriter.setInsertionPoint(wiresOp);
          // Create the NotLibOp at function level for now
          auto conditionType = condition.getType();
          auto invertedCondition = rewriter.create<calyx::NotLibOp>(
              ifOp.getLoc(), "inverted_cond_" + getOpUniqueName(ifOp),
              llvm::SmallVector<mlir::Type>{conditionType, conditionType});
          rewriter.setInsertionPoint(&op);
          rewriter.create<calyx::AssignOp>(
              ifOp.getLoc(), invertedCondition.getIn(), condition);
          rewriter.setInsertionPoint(&op);
          // Create assignment: reg.in = elseYieldValue
          rewriter.create<calyx::AssignOp>(op.getLoc(), resultReg.getIn(),
                                           yieldValue,
                                           invertedCondition.getOut());
          // Create assignment: reg.write_en = 1
          rewriter.create<calyx::AssignOp>(op.getLoc(), resultReg.getWriteEn(),
                                           constantOne,
                                           invertedCondition.getOut());
          // Create group_done
          rewriter.create<calyx::GroupDoneOp>(op.getLoc(), resultReg.getDone());
        }
        // Erase the yield op after processing
        rewriter.eraseOp(yieldOp);
      }
    }
  }
  // Replace the scf.if with the calyx.if, providing the register output as the
  // result
  for (size_t i = 0; i < ifResultsRegs.size(); ++i) {
    // Connect the calyx.if result to the register output
    ifOp.getResult(i).replaceAllUsesWith(ifResultsRegs[i].getOut());
  }

  // Now replace the scf.if with calyx.if
  rewriter.setInsertionPoint(ifOp);
  auto calyxIfOp = rewriter.create<calyx::IfOp>(ifOp.getLoc(), condition,
                                                nullptr, !elseRegion.empty());

  // Move the then region operations to calyx.if then region
  if (!thenRegion.empty()) {
    auto &calyxThenBlock = calyxIfOp.getThenRegion().front();
    auto &sourceThenBlock = thenRegion.front();
    calyxThenBlock.getOperations().splice(calyxThenBlock.begin(),
                                          sourceThenBlock.getOperations());
  }

  // Move the else region operations to calyx.if else region (if it exists)
  if (!elseRegion.empty()) {
    auto &calyxElseBlock = calyxIfOp.getElseRegion().front();
    auto &sourceElseBlock = elseRegion.front();
    calyxElseBlock.getOperations().splice(calyxElseBlock.begin(),
                                          sourceElseBlock.getOperations());
  }

  rewriter.eraseOp(ifOp);
  return success();
}

// SCF for to Calyx conversion pattern implementation
LogicalResult ScfForToCalyxPattern::matchAndRewrite(
    mlir::scf::ForOp forOp, mlir::scf::ForOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // SCF patterns run AFTER function-to-component conversion, so we should be in
  // a component context
  auto componentOp = forOp->getParentOfType<calyx::ComponentOp>();
  if (!componentOp) {
    return rewriter.notifyMatchFailure(forOp, "scf.for not within a component");
  }

  calyx::WiresOp wiresOp = componentOp.getWiresOp();
  if (!wiresOp) {
    return rewriter.notifyMatchFailure(forOp, "No WiresOp found in component");
  }

  // Use the enhanced SCF for conversion logic directly instead of the old
  // transformScfForToCalyx Extract constant bounds
  auto lbConstant =
      forOp.getLowerBound().getDefiningOp<mlir::arith::ConstantOp>();
  auto ubConstant =
      forOp.getUpperBound().getDefiningOp<mlir::arith::ConstantOp>();
  auto stepConstant = forOp.getStep().getDefiningOp<mlir::arith::ConstantOp>();

  if (!lbConstant || !ubConstant || !stepConstant) {
    return rewriter.notifyMatchFailure(
        forOp, "Only constant-bound for loops are supported, convert the for "
               "loop to while loop and lower via the while op");
  }

  // Extract constant values
  auto lbIntAttr = mlir::dyn_cast<mlir::IntegerAttr>(lbConstant.getValue());
  auto ubIntAttr = mlir::dyn_cast<mlir::IntegerAttr>(ubConstant.getValue());
  auto stepIntAttr = mlir::dyn_cast<mlir::IntegerAttr>(stepConstant.getValue());

  if (!lbIntAttr || !ubIntAttr || !stepIntAttr) {
    return rewriter.notifyMatchFailure(
        forOp, "Non-integer constant bounds in for loop");
  }

  auto lbValue = lbIntAttr.getInt();
  auto ubValue = ubIntAttr.getInt();
  auto stepValue = stepIntAttr.getInt();

  // Calculate iteration count
  if (stepValue <= 0 || ubValue <= lbValue) {
    return rewriter.notifyMatchFailure(forOp, "Invalid for loop bounds");
  }

  int64_t count = (ubValue - lbValue + stepValue - 1) / stepValue;

  // Get block arguments from the for loop body
  Value inductionVar;
  llvm::SmallVector<Value> iterationArgs;
  if (!forOp.getRegion().empty() &&
      !forOp.getRegion().front().getArguments().empty()) {
    auto &forBlock = forOp.getRegion().front();
    auto blockArgs = forBlock.getArguments();

    // First argument is always the induction variable
    if (!blockArgs.empty()) {
      inductionVar = blockArgs[0];
    }

    // Remaining arguments are iteration arguments (iter_args)
    for (unsigned i = 1; i < blockArgs.size(); ++i) {
      iterationArgs.push_back(blockArgs[i]);
    }
  }

  // Determine bit width for constants - handle index type by using 32-bit
  // default
  unsigned bitWidth = 32; // Default for index type
  if (inductionVar) {
    // For unused utility function, just use i32 as default
    Type convertedType = IntegerType::get(forOp.getContext(), 32);
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(convertedType)) {
      bitWidth = intType.getWidth();
    }
  }

  // Create a counter register to replace the induction variable
  OpBuilder componentBuilder(wiresOp);
  auto &wiresBlock = wiresOp.getBodyRegion().front();
  OpBuilder wiresBuilder(&wiresBlock, wiresBlock.end());

  Value counterReg;
  if (inductionVar) {
    std::string counterName = "for_counter_" + getOpUniqueName(forOp);
    // Use type converter to convert index to i32
    Type convertedType =
        this->getTypeConverter()->convertType(inductionVar.getType());
    auto counterRegOp = componentBuilder.create<calyx::RegisterOp>(
        forOp.getLoc(), counterName, convertedType);
    counterReg = counterRegOp.getOut();
  }

  // Create registers for all iteration arguments
  llvm::SmallVector<Value> iterArgRegs;
  for (unsigned i = 0; i < iterationArgs.size(); ++i) {
    std::string iterArgName =
        "for_iter_arg_" + std::to_string(i) + "_" + getOpUniqueName(forOp);
    auto iterArgRegOp = componentBuilder.create<calyx::RegisterOp>(
        forOp.getLoc(), iterArgName, iterationArgs[i].getType());
    iterArgRegs.push_back(iterArgRegOp.getOut());
  }

  // Initialize counter register
  if (counterReg && inductionVar) {
    auto initConstant =
        getOrCreateConstant(componentOp.getOperation(), lbValue, bitWidth);
    auto counterRegOp = counterReg.getDefiningOp<calyx::RegisterOp>();
    if (counterRegOp) {
      rewriter.create<calyx::AssignOp>(forOp.getLoc(), counterRegOp.getIn(),
                                       initConstant);
      rewriter.create<calyx::AssignOp>(
          forOp.getLoc(), counterRegOp.getWriteEn(),
          getOrCreateConstant(componentOp.getOperation(), 1, 1));
    }
  }

  // Initialize iteration argument registers with their initial values
  auto initArgs = forOp.getInitArgs();
  for (unsigned i = 0; i < iterArgRegs.size() && i < initArgs.size(); ++i) {
    auto iterArgRegOp = iterArgRegs[i].getDefiningOp<calyx::RegisterOp>();
    if (iterArgRegOp) {
      rewriter.create<calyx::AssignOp>(forOp.getLoc(), iterArgRegOp.getIn(),
                                       initArgs[i]);
      rewriter.create<calyx::AssignOp>(
          forOp.getLoc(), iterArgRegOp.getWriteEn(),
          getOrCreateConstant(componentOp.getOperation(), 1, 1));
    }
  }

  // Add a single group done signal to the initialization group.
  // Prefer iteration argument register done signals. If multiple iter args,
  // AND all of their done signals together. Fallback to counterReg if none.
  auto buildAndChain = [&](SmallVector<Value> &signals) -> Value {
    if (signals.empty())
      return nullptr;
    if (signals.size() == 1)
      return signals.front();
    // Chain AndLibOps pairwise: (((s0 & s1) & s2) & ...)
    Value accum = signals[0];
    for (size_t i = 1; i < signals.size(); i++) {
      auto loc = forOp.getLoc();
      // Create AndLibOp at component level before wires.
      OpBuilder compBuilder(componentOp.getContext());
      compBuilder.setInsertionPoint(wiresOp);
      SmallVector<Type> andTypes = {rewriter.getI1Type(), rewriter.getI1Type(),
                                    rewriter.getI1Type()};
      auto andOp = compBuilder.create<calyx::AndLibOp>(
          loc,
          rewriter.getStringAttr("for_iter_args_and_" + getOpUniqueName(forOp) +
                                 "_" + std::to_string(i)),
          andTypes);
      // Connect assigns in wires region end.
      auto &wiresBlockRef = wiresOp.getBodyRegion().front();
      OpBuilder wiresAssignBuilder(&wiresBlockRef, wiresBlockRef.end());
      wiresAssignBuilder.create<calyx::AssignOp>(loc, andOp.getLeft(), accum);
      wiresAssignBuilder.create<calyx::AssignOp>(loc, andOp.getRight(),
                                                 signals[i]);
      accum = andOp.getOut();
    }
    return accum;
  };

  Value initDoneSignal = nullptr;
  if (!iterArgRegs.empty()) {
    SmallVector<Value> iterDoneSignals;
    for (auto v : iterArgRegs) {
      if (auto regOp = v.getDefiningOp<calyx::RegisterOp>())
        iterDoneSignals.push_back(regOp.getDone());
    }
    initDoneSignal = buildAndChain(iterDoneSignals);
  } else if (counterReg) {
    if (auto counterRegOp = counterReg.getDefiningOp<calyx::RegisterOp>())
      initDoneSignal = counterRegOp.getDone();
  }
  if (initDoneSignal)
    rewriter.create<calyx::GroupDoneOp>(forOp.getLoc(), initDoneSignal);

  // Create the calyx.repeat operation
  auto calyxRepeatOp = rewriter.create<calyx::RepeatOp>(
      forOp.getLoc(), static_cast<uint32_t>(count));

  // Process the for loop body
  auto &forRegion = forOp.getRegion();
  if (!forRegion.empty()) {
    auto &forBlock = forRegion.front();
    auto &repeatBlock = calyxRepeatOp.getBodyRegion().front();

    // STEP 1: Replace all uses of block arguments with their corresponding
    // registers
    if (inductionVar && counterReg) {
      replaceAllUsesInRegionWith(inductionVar, counterReg, forRegion);
    }

    for (unsigned i = 0; i < iterationArgs.size(); ++i) {
      if (i < iterArgRegs.size()) {
        replaceAllUsesInRegionWith(iterationArgs[i], iterArgRegs[i], forRegion);
      }
    }

    // STEP 2: Extract yield values before moving operations
    llvm::SmallVector<Value> yieldValues;
    mlir::scf::YieldOp yieldOpFound = nullptr;
    for (auto &op : forBlock) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(op)) {
        yieldOpFound = yieldOp;
        for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
          yieldValues.push_back(yieldOp.getOperand(i));
        }
        break; // Only one yield expected
      }
    }

    // STEP 3: Move all operations except the yield
    for (auto &op : llvm::make_early_inc_range(forBlock)) {
      if (isa<mlir::scf::YieldOp>(op)) {
        // Replace the yield with a GroupDoneOp similar to ifOp lowering.
        // Prefer the counter register done signal, else first iter arg reg.
        // Prefer iter arg done signals. If multiple, AND them. Fallback to
        // counterReg.
        Value doneSignal;
        if (!iterArgRegs.empty()) {
          SmallVector<Value> iterDoneSignals;
          for (auto v : iterArgRegs)
            if (auto regOp = v.getDefiningOp<calyx::RegisterOp>())
              iterDoneSignals.push_back(regOp.getDone());
          doneSignal = buildAndChain(iterDoneSignals);
        }

        if (doneSignal)
          rewriter.create<calyx::GroupDoneOp>(op.getLoc(), doneSignal);
        rewriter.eraseOp(&op); // Remove yield; Calyx repeat uses done signal
        continue;
      }
      op.moveBefore(&repeatBlock, repeatBlock.end());
    }

    // Add update logic at the end of repeat body
    OpBuilder repeatBuilder(&repeatBlock, repeatBlock.end());

    // Update iteration argument registers with yield values
    for (unsigned i = 0; i < iterArgRegs.size() && i < yieldValues.size();
         ++i) {
      auto iterArgRegOp = iterArgRegs[i].getDefiningOp<calyx::RegisterOp>();
      if (iterArgRegOp && yieldValues[i]) {
        repeatBuilder.create<calyx::AssignOp>(
            forOp.getLoc(), iterArgRegOp.getIn(), yieldValues[i]);
        repeatBuilder.create<calyx::AssignOp>(
            forOp.getLoc(), iterArgRegOp.getWriteEn(),
            getOrCreateConstant(componentOp.getOperation(), 1, 1));
        // Done signal handled by repeat operation
      }
    }

    // If we created a counter, add increment logic at the end of repeat body
    if (counterReg && inductionVar) {
      // Create step constant
      auto stepConstantValue =
          getOrCreateConstant(componentOp.getOperation(), stepValue, bitWidth);

      // Create add operation for counter increment
      std::string addOpName = "for_counter_add_" + getOpUniqueName(forOp);
      // Use converted type for add operation
      Type convertedType =
          this->getTypeConverter()->convertType(inductionVar.getType());
      SmallVector<Type> addResultTypes = {convertedType, convertedType,
                                          convertedType};

      auto addOp = componentBuilder.create<calyx::AddLibOp>(
          forOp.getLoc(), rewriter.getStringAttr(addOpName), addResultTypes);

      // Connect the add operation inputs in wires
      wiresBuilder.create<calyx::AssignOp>(forOp.getLoc(), addOp.getLeft(),
                                           counterReg);
      wiresBuilder.create<calyx::AssignOp>(forOp.getLoc(), addOp.getRight(),
                                           stepConstantValue);

      // Assign the incremented value back to the counter register in the repeat
      // body
      auto counterRegOp = counterReg.getDefiningOp<calyx::RegisterOp>();
      if (counterRegOp) {
        repeatBuilder.create<calyx::AssignOp>(
            forOp.getLoc(), counterRegOp.getIn(), addOp.getOut());
        repeatBuilder.create<calyx::AssignOp>(
            forOp.getLoc(), counterRegOp.getWriteEn(),
            getOrCreateConstant(componentOp.getOperation(), 1, 1));
        // Done signal handled by repeat operation
      }
    }
  }

  // If the for loop had results, replace them with the final iteration argument
  // register values
  for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
    auto forResult = forOp.getResult(i);

    if (i < iterArgRegs.size()) {
      // Replace all uses of the for result with the iteration argument register
      // output
      forResult.replaceAllUsesWith(iterArgRegs[i]);
    } else {
      // Fallback: create a temporary register if we don't have enough iter arg
      // registers
      rewriter.setInsertionPoint(wiresOp);
      std::string regName =
          "for_result_" + std::to_string(i) + "_" + getOpUniqueName(forOp);
      auto resultReg = rewriter.create<calyx::RegisterOp>(
          forOp.getLoc(), regName, forResult.getType());
      forResult.replaceAllUsesWith(resultReg.getOut());
    }
  }

  // Erase the original for loop
  rewriter.eraseOp(forOp);

  return success();
}

// Helper function to check if a group only contains done signals
static bool groupOnlyHasDone(calyx::GroupOp groupOp) {
  auto *groupBlock = groupOp.getBodyBlock();
  if (!groupBlock)
    return false;

  // Count non-done operations
  size_t nonDoneOpsCount = 0;
  bool hasDone = false;

  for (auto &op : *groupBlock) {
    if (isa<calyx::GroupDoneOp>(op)) {
      hasDone = true;
    } else {
      // Count any other operations (assignments, etc.)
      nonDoneOpsCount++;
    }
  }

  // Group only has done if it has a done signal and no other operations
  return hasDone && (nonDoneOpsCount == 0);
}

// Helper function to find the group referenced by an enable operation
static calyx::GroupOp findReferencedGroup(calyx::EnableOp enableOp,
                                          calyx::ComponentOp componentOp) {
  StringRef groupName = enableOp.getGroupName();

  // Search for the group in the component's wires section
  auto wiresOp = componentOp.getWiresOp();
  if (!wiresOp)
    return nullptr;

  for (auto groupOp : wiresOp.getOps<calyx::GroupOp>()) {
    if (groupOp.getSymName() == groupName) {
      return groupOp;
    }
  }

  return nullptr;
}

// Control flow wrapping pattern implementation
LogicalResult ControlFlowWrappingPattern::matchAndRewrite(
    calyx::ControlOp controlOp, calyx::ControlOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto &controlRegion = controlOp.getBodyRegion();
  if (controlRegion.empty())
    return failure();

  auto &controlBlock = controlRegion.front();
  if (controlBlock.empty())
    return failure();

  bool changed = false;

  auto wrapRegionWithSeq = [&](Region &region) {
    if (region.empty())
      return false;
    Block &block = region.front();
    if (block.empty())
      return false;
    // If already a single seq, skip
    if (std::next(block.begin()) == block.end() &&
        isa<calyx::SeqOp>(block.front()))
      return false;
    // Collect current ops to move before creating the seq op
    SmallVector<Operation *> opsToMove;
    for (Operation &op : block)
      opsToMove.push_back(&op);

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(&block);
    auto seq = rewriter.create<calyx::SeqOp>(opsToMove.front()->getLoc());
    auto &seqBlock = seq.getBodyRegion().front();
    // Move only the previously collected ops into the seq block
    for (Operation *op : opsToMove)
      op->moveBefore(&seqBlock, seqBlock.end());
    return true;
  };

  // Iterate top-level control ops and wrap their regions
  for (Operation &op : llvm::make_early_inc_range(controlBlock)) {
    if (auto ifOp = dyn_cast<calyx::IfOp>(&op)) {
      changed |= wrapRegionWithSeq(ifOp.getThenRegion());
      if (!ifOp.getElseRegion().empty())
        changed |= wrapRegionWithSeq(ifOp.getElseRegion());
      continue;
    }
    if (auto rep = dyn_cast<calyx::RepeatOp>(&op)) {
      changed |= wrapRegionWithSeq(rep.getBodyRegion());
      continue;
    }
    if (auto wh = dyn_cast<calyx::WhileOp>(&op)) {
      changed |= wrapRegionWithSeq(wh.getBodyRegion());
      continue;
    }
    // For nested seq/par, we don't need to wrap; seq is already sequential.
    // Optionally, could wrap bodies of par regions similarly if desired.
  }

  return changed ? success() : failure();
}

// Empty group optimization pattern implementation
LogicalResult EmptyGroupOptimizationPattern::matchAndRewrite(
    calyx::ComponentOp componentOp, calyx::ComponentOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Find the wires section to access groups
  auto wiresOp = componentOp.getWiresOp();
  if (!wiresOp) {
    return failure(); // No wires section
  }

  // Find the control section to access enables
  auto controlOp = componentOp.getControlOp();
  if (!controlOp) {
    return failure(); // No control section
  }

  auto &controlRegion = controlOp.getBodyRegion();
  if (controlRegion.empty()) {
    return failure(); // Empty control region
  }

  auto &controlBlock = controlRegion.front();

  // Collect groups that only have done signals and their corresponding enables
  SmallVector<calyx::GroupOp> groupsToRemove;
  SmallVector<calyx::EnableOp> enablesToRemove;

  // First, identify groups that only have done signals
  for (auto groupOp : wiresOp.getOps<calyx::GroupOp>()) {
    if (groupOnlyHasDone(groupOp)) {
      groupsToRemove.push_back(groupOp);

      // Find enables that reference this group
      StringRef groupName = groupOp.getSymName();
      controlBlock.walk([&](calyx::EnableOp enableOp) {
        if (enableOp.getGroupName() == groupName) {
          enablesToRemove.push_back(enableOp);
        }
      });
    }
  }

  if (groupsToRemove.empty()) {
    return failure(); // No empty groups found
  }

  // Remove the enables first
  for (auto enableOp : enablesToRemove) {
    rewriter.eraseOp(enableOp);
  }

  // Then remove the groups
  for (auto groupOp : groupsToRemove) {
    rewriter.eraseOp(groupOp);
  }

  return success();
}

/// Transform a SCF For operation to Calyx hardware constructs
/// This function creates registers and hardware for constant-bound for loops
LogicalResult transformScfForToCalyx(mlir::scf::ForOp forOp,
                                     OpBuilder &wiresBuilder,
                                     OpBuilder &functionBuilder) {
  // Check if this is a constant-bound for loop
  auto lbConstant =
      forOp.getLowerBound().getDefiningOp<mlir::arith::ConstantOp>();
  auto ubConstant =
      forOp.getUpperBound().getDefiningOp<mlir::arith::ConstantOp>();
  auto stepConstant = forOp.getStep().getDefiningOp<mlir::arith::ConstantOp>();

  if (!lbConstant || !ubConstant || !stepConstant) {
    return forOp.emitError("Only constant-bound for loops are supported");
  }

  // Extract constant values
  auto lbIntAttr = mlir::dyn_cast<mlir::IntegerAttr>(lbConstant.getValue());
  auto ubIntAttr = mlir::dyn_cast<mlir::IntegerAttr>(ubConstant.getValue());
  auto stepIntAttr = mlir::dyn_cast<mlir::IntegerAttr>(stepConstant.getValue());

  if (!lbIntAttr || !ubIntAttr || !stepIntAttr) {
    return forOp.emitError("Non-integer constant bounds in for loop");
  }

  auto lbValue = lbIntAttr.getInt();
  auto ubValue = ubIntAttr.getInt();
  auto stepValue = stepIntAttr.getInt();

  // Calculate iteration count
  if (stepValue <= 0 || ubValue <= lbValue) {
    return forOp.emitError("Invalid for loop bounds");
  }

  int64_t count = (ubValue - lbValue + stepValue - 1) / stepValue;

  // Validate that we're inside a function (for context checking)
  auto funcOp = forOp->getParentOfType<mlir::func::FuncOp>();
  if (!funcOp) {
    return forOp.emitError("scf.for not within a function");
  }

  // Handle the result value if the for loop has results
  Value forResult;
  if (forOp.getNumResults() > 0) {
    forResult = forOp.getResult(0);
  }

  // Get block arguments from the for loop body
  Value inductionVar;
  llvm::SmallVector<Value> iterationArgs;
  if (!forOp.getRegion().empty() &&
      !forOp.getRegion().front().getArguments().empty()) {
    auto &forBlock = forOp.getRegion().front();
    auto blockArgs = forBlock.getArguments();

    // First argument is always the induction variable
    if (!blockArgs.empty()) {
      inductionVar = blockArgs[0];
    }

    // Remaining arguments are iteration arguments (iter_args)
    for (unsigned i = 1; i < blockArgs.size(); ++i) {
      iterationArgs.push_back(blockArgs[i]);
    }
  }

  // Determine bit width for constants - handle index type by using 32-bit
  // default
  unsigned bitWidth = 32; // Default for index type
  if (inductionVar) {
    // For unused utility function, just use i32 as default
    Type convertedType = IntegerType::get(forOp.getContext(), 32);
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(convertedType)) {
      bitWidth = intType.getWidth();
    }
  }

  // Find the wires operation for register creation
  auto &entryBlock = funcOp.front();
  auto wiresOps = entryBlock.getOps<calyx::WiresOp>();
  if (wiresOps.empty()) {
    return forOp.emitError("No WiresOp found in function");
  }
  auto wiresOp = *wiresOps.begin();

  // Create a counter register to replace the induction variable
  Value counterReg;
  if (inductionVar) {
    functionBuilder.setInsertionPoint(wiresOp);

    // Create a counter register with the same type as the induction variable
    std::string counterName = "for_counter_" + getOpUniqueName(forOp);
    // Use default i32 type for index in utility function
    Type convertedType = inductionVar.getType().isIndex()
                             ? IntegerType::get(forOp.getContext(), 32)
                             : inductionVar.getType();
    auto counterRegOp = functionBuilder.create<calyx::RegisterOp>(
        forOp.getLoc(), counterName, convertedType);
    counterReg = counterRegOp.getOut();

    // llvm::errs() << "Created counter register with type: " <<
    // counterReg.getType() << "\n";

    // We'll handle counter initialization in a separate initialization group
    // rather than continuous assignment to avoid conflicts
  }

  // Create registers for all iteration arguments
  llvm::SmallVector<Value> iterArgRegs;
  for (unsigned i = 0; i < iterationArgs.size(); ++i) {
    functionBuilder.setInsertionPoint(wiresOp);

    std::string iterArgName =
        "for_iter_arg_" + std::to_string(i) + "_" + getOpUniqueName(forOp);
    auto iterArgRegOp = functionBuilder.create<calyx::RegisterOp>(
        forOp.getLoc(), iterArgName, iterationArgs[i].getType());
    iterArgRegs.push_back(iterArgRegOp.getOut());

    // Store the initial value - we'll handle initialization differently
    // since we can't have continuous assignments and conditional assignments
    // to the same register
  }

  // Replace with calyx.repeat
  OpBuilder builder(forOp);

  // Initialize counter register before the repeat
  if (counterReg && inductionVar) {
    auto initConstant =
        getOrCreateConstant(funcOp.getOperation(), lbValue, bitWidth);
    auto counterRegOp = counterReg.getDefiningOp<calyx::RegisterOp>();
    if (counterRegOp) {
      builder.create<calyx::AssignOp>(forOp.getLoc(), counterRegOp.getIn(),
                                      initConstant);
      builder.create<calyx::AssignOp>(
          forOp.getLoc(), counterRegOp.getWriteEn(),
          getOrCreateConstant(funcOp.getOperation(), 1, 1));
    }
  }

  // Initialize iteration argument registers before the repeat
  auto initArgs = forOp.getInitArgs();
  for (unsigned i = 0; i < iterArgRegs.size() && i < initArgs.size(); ++i) {
    auto iterArgRegOp = iterArgRegs[i].getDefiningOp<calyx::RegisterOp>();
    if (iterArgRegOp) {
      builder.create<calyx::AssignOp>(forOp.getLoc(), iterArgRegOp.getIn(),
                                      initArgs[i]);
      builder.create<calyx::AssignOp>(
          forOp.getLoc(), iterArgRegOp.getWriteEn(),
          getOrCreateConstant(funcOp.getOperation(), 1, 1));
    }
  }

  auto calyxRepeatOp = builder.create<calyx::RepeatOp>(
      forOp.getLoc(), static_cast<uint32_t>(count));

  // Do all replacements FIRST, then do all moves
  auto &forRegion = forOp.getRegion();
  if (!forRegion.empty()) {
    auto &forBlock = forRegion.front();
    auto &repeatBlock = calyxRepeatOp.getBodyRegion().front();

    // STEP 1: Replace all uses of block arguments with their corresponding
    // registers Do this BEFORE moving any operations to avoid use-after-move
    // issues

    // Replace induction variable with counter register
    if (inductionVar && counterReg) {
      replaceAllUsesInRegionWith(inductionVar, counterReg, forRegion);
    }

    // Replace all iteration arguments with their corresponding registers
    for (unsigned i = 0; i < iterationArgs.size(); ++i) {
      if (i < iterArgRegs.size()) {
        replaceAllUsesInRegionWith(iterationArgs[i], iterArgRegs[i], forRegion);
      }
    }

    // STEP 2: Extract yield values before moving operations (for iteration arg
    // updates)
    llvm::SmallVector<Value> yieldValues;
    for (auto &op : forBlock) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(op)) {
        for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
          yieldValues.push_back(yieldOp.getOperand(i));
        }
        break;
      }
    }

    // STEP 3: Now it's safe to move all operations except the yield
    for (auto &op : llvm::make_early_inc_range(forBlock)) {
      if (isa<mlir::scf::YieldOp>(op)) {
        op.erase(); // Remove yield as Calyx repeat doesn't need it
        continue;
      }
      op.moveBefore(&repeatBlock, repeatBlock.end());
    }

    // Add update logic at the end of repeat body
    OpBuilder repeatBuilder(&repeatBlock, repeatBlock.end());

    // Update iteration argument registers with yield values
    for (unsigned i = 0; i < iterArgRegs.size() && i < yieldValues.size();
         ++i) {
      auto iterArgRegOp = iterArgRegs[i].getDefiningOp<calyx::RegisterOp>();
      if (iterArgRegOp && yieldValues[i]) {
        // Update iteration argument register with yield value
        repeatBuilder.create<calyx::AssignOp>(
            forOp.getLoc(), iterArgRegOp.getIn(), yieldValues[i]);
        repeatBuilder.create<calyx::AssignOp>(
            forOp.getLoc(), iterArgRegOp.getWriteEn(),
            getOrCreateConstant(funcOp.getOperation(), 1, 1));
      }
    }

    // If we created a counter, add increment logic at the end of repeat body
    if (counterReg && inductionVar) {
      // Create step constant using getOrCreateConstant
      auto stepConstantValue =
          getOrCreateConstant(funcOp.getOperation(), stepValue, bitWidth);

      // Create add operation for counter increment
      // Need to create AddLibOp as a component-level operation
      auto componentOp = forOp->getParentOfType<calyx::ComponentOp>();
      if (componentOp) {
        auto wiresOp = componentOp.getWiresOp();
        OpBuilder componentBuilder(wiresOp);

        std::string addOpName = "for_counter_add_" + getOpUniqueName(forOp);
        // Use converted type for add operation
        // For unused utility function, just use i32 as default
        Type convertedType = IntegerType::get(forOp.getContext(), 32);
        SmallVector<Type> addResultTypes = {convertedType, convertedType,
                                            convertedType};
        // Create AddLibOp for counter increment
        auto addOp = componentBuilder.create<calyx::AddLibOp>(
            forOp.getLoc(), componentBuilder.getStringAttr(addOpName),
            addResultTypes);

        // Connect the add operation inputs in wires
        auto &wiresBlock = wiresOp.getBodyRegion().front();
        OpBuilder addWiresBuilder(&wiresBlock, wiresBlock.end());
        addWiresBuilder.create<calyx::AssignOp>(forOp.getLoc(), addOp.getLeft(),
                                                counterReg);
        addWiresBuilder.create<calyx::AssignOp>(
            forOp.getLoc(), addOp.getRight(), stepConstantValue);

        // Assign the incremented value back to the counter register in the
        // repeat body
        auto counterRegOp = counterReg.getDefiningOp<calyx::RegisterOp>();
        if (counterRegOp) {
          // Make sure we're using the correct addOp output (should be index
          // type)
          repeatBuilder.create<calyx::AssignOp>(
              forOp.getLoc(), counterRegOp.getIn(), addOp.getOut());
          repeatBuilder.create<calyx::AssignOp>(
              forOp.getLoc(), counterRegOp.getWriteEn(),
              getOrCreateConstant(funcOp.getOperation(), 1, 1));
        }
      }
    }
  }

  // If the for loop had results, replace them with the final iteration argument
  // register values
  for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
    auto forResult = forOp.getResult(i);

    // The for loop result corresponds to the final value of the i-th iteration
    // argument
    if (i < iterArgRegs.size()) {
      // Replace all uses of the for result with the iteration argument register
      // output
      forResult.replaceAllUsesWith(iterArgRegs[i]);
    } else {
      // Fallback: create a temporary register if we don't have enough iter arg
      // registers
      auto &entryBlock = funcOp.front();
      auto wiresOps = entryBlock.getOps<calyx::WiresOp>();
      if (!wiresOps.empty()) {
        auto wiresOp = *wiresOps.begin();
        functionBuilder.setInsertionPoint(wiresOp);
      }

      std::string regName =
          "for_result_" + std::to_string(i) + "_" + getOpUniqueName(forOp);
      auto resultReg = functionBuilder.create<calyx::RegisterOp>(
          forOp.getLoc(), regName, forResult.getType());

      forResult.replaceAllUsesWith(resultReg.getOut());
    }
  }

  // Erase the original for loop
  forOp.erase();

  return success();
}

/// Transform a SCF While operation to Calyx hardware constructs
/// This function creates registers and hardware for while loops
LogicalResult transformScfWhileToCalyx(mlir::scf::WhileOp whileOp,
                                       OpBuilder &wiresBuilder,
                                       OpBuilder &functionBuilder) {
  // Get the condition from the before region
  Block *beforeBlock = whileOp.getBefore().getBlocks().empty()
                           ? nullptr
                           : &whileOp.getBefore().front();
  if (!beforeBlock) {
    return whileOp.emitError("while op has no before region");
  }

  // Find the condition operation in the before block
  Value condition = nullptr;
  for (auto &beforeOp : beforeBlock->getOperations()) {
    if (auto condOp = dyn_cast<mlir::scf::ConditionOp>(beforeOp)) {
      condition = condOp.getCondition();
      break;
    }
  }
  if (!condition) {
    return whileOp.emitError("could not find condition in while op");
  }

  // Create a new Calyx WhileOp with the condition
  OpBuilder builder(whileOp);
  auto calyxWhileOp =
      builder.create<calyx::WhileOp>(whileOp.getLoc(), condition, nullptr);

  // Move the after region (body) contents to the new WhileOp
  if (!whileOp.getAfter().empty()) {
    auto &afterBlock = whileOp.getAfter().front();
    auto &whileBodyBlock = calyxWhileOp.getBodyRegion().front();

    // Move all operations except yield
    for (auto &op : llvm::make_early_inc_range(afterBlock)) {
      if (isa<mlir::scf::YieldOp>(op)) {
        op.erase(); // Remove yield as Calyx while doesn't need it
        continue;
      }
      op.moveBefore(&whileBodyBlock, whileBodyBlock.end());
    }
  }

  // Also need to move the condition evaluation from before region
  // Move operations from before region (except condition op) to before the
  // while
  for (auto &op : llvm::make_early_inc_range(*beforeBlock)) {
    if (isa<mlir::scf::ConditionOp>(op)) {
      op.erase(); // Remove condition op
      continue;
    }
    op.moveBefore(calyxWhileOp);
  }

  // Erase the original while loop
  whileOp.erase();

  return success();
}

} // namespace lowertocalyx
} // namespace circt
