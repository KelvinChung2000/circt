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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cassert>

using namespace circt;
using namespace circt::lowertocalyx;
using namespace mlir;

// Forward declaration from LowerToCalyx.cpp
std::string getOpUniqueName(mlir::Operation *op);

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
LogicalResult transformScfIfToCalyx(mlir::scf::IfOp ifOp,
                                    OpBuilder &wiresBuilder,
                                    OpBuilder &functionBuilder) {
  // Skip if this if operation doesn't have results
  if (ifOp.getNumResults() == 0) {
    return success();
  }

  // Get the condition and create register for this if operation
  Value condition = ifOp.getCondition();
  Value ifResult = ifOp.getResult(0);

  // Create register at function body level - before wires op
  // Find the wires operation and insert before it
  auto funcOp = ifOp->getParentOfType<mlir::func::FuncOp>();
  if (!funcOp) {
    return ifOp.emitError("scf.if not within a function");
  }

  auto &entryBlock = funcOp.front();
  auto wiresOps = entryBlock.getOps<calyx::WiresOp>();
  if (wiresOps.empty()) {
    return ifOp.emitError("No WiresOp found in function");
  }
  auto wiresOp = *wiresOps.begin();

  // Set insertion point before wires op
  functionBuilder.setInsertionPoint(wiresOp);

  std::string regName = "reg_" + getOpUniqueName(ifOp);
  auto resultReg = functionBuilder.create<calyx::RegisterOp>(
      ifOp.getLoc(), regName, ifResult.getType());

  // Value regIn = resultReg.getResult(0);      // reg.in
  // Value regWriteEn = resultReg.getResult(1); // reg.write_en
  // Value regOut = resultReg.getResult(4);     // reg.out
  // Value regDone = resultReg.getResult(5);    // reg.done

  // Find the yield values in then and else regions
  Value thenYieldValue = nullptr;
  Value elseYieldValue = nullptr;

  // Get then yield value
  auto &thenRegion = ifOp.getThenRegion();
  if (!thenRegion.empty()) {
    auto &thenBlock = thenRegion.front();
    for (auto &op : thenBlock) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(op)) {
        if (yieldOp.getNumOperands() > 0) {
          thenYieldValue = yieldOp.getOperand(0);
        }
        break;
      }
    }
  }

  // Get else yield value
  auto &elseRegion = ifOp.getElseRegion();
  if (!elseRegion.empty()) {
    auto &elseBlock = elseRegion.front();
    for (auto &op : elseBlock) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(op)) {
        if (yieldOp.getNumOperands() > 0) {
          elseYieldValue = yieldOp.getOperand(0);
        }
        break;
      }
    }
  }

  if (!thenYieldValue || !elseYieldValue) {
    return ifOp.emitError("Could not find yield values in scf.if branches");
  }

  // Check if both branches have no side effects - if so, use MuxLibOp
  // optimization
  bool thenHasNoSideEffects = hasNoSideEffects(ifOp.getThenRegion());
  bool elseHasNoSideEffects = hasNoSideEffects(ifOp.getElseRegion());

  if (thenHasNoSideEffects && elseHasNoSideEffects) {
    // Both branches are side-effect-free, use MuxLibOp instead
    std::string muxName = "mux_" + getOpUniqueName(ifOp);
    auto resultType = ifResult.getType();

    // First, move all operations from both regions out of the scf.if
    OpBuilder builder(ifOp);
    IRMapping thenMapping, elseMapping;

    // Clone operations from then region (excluding yield)
    Value thenResult = nullptr;
    if (!ifOp.getThenRegion().empty()) {
      auto &thenBlock = ifOp.getThenRegion().front();
      for (auto &op : llvm::make_early_inc_range(thenBlock)) {
        if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(op)) {
          if (yieldOp.getNumOperands() > 0) {
            thenResult = thenMapping.lookupOrDefault(yieldOp.getOperand(0));
            if (!thenResult)
              thenResult = yieldOp.getOperand(0);
          }
          continue; // Don't clone yield
        }
        // Clone the operation before the scf.if
        auto *clonedOp = builder.clone(op, thenMapping);
        (void)clonedOp; // Mark as used
      }
    }

    // Clone operations from else region (excluding yield)
    Value elseResult = nullptr;
    if (!ifOp.getElseRegion().empty()) {
      auto &elseBlock = ifOp.getElseRegion().front();
      for (auto &op : llvm::make_early_inc_range(elseBlock)) {
        if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(op)) {
          if (yieldOp.getNumOperands() > 0) {
            elseResult = elseMapping.lookupOrDefault(yieldOp.getOperand(0));
            if (!elseResult)
              elseResult = yieldOp.getOperand(0);
          }
          continue; // Don't clone yield
        }
        // Clone the operation before the scf.if
        auto *clonedOp = builder.clone(op, elseMapping);
        (void)clonedOp; // Mark as used
      }
    }

    // Create MuxLibOp at function level
    functionBuilder.setInsertionPoint(wiresOp);
    auto muxOp = functionBuilder.create<calyx::MuxLibOp>(
        ifOp.getLoc(), muxName,
        llvm::SmallVector<mlir::Type>{condition.getType(), resultType,
                                      resultType, resultType});

    // Set up the mux connections in wires
    wiresBuilder.setInsertionPointToEnd(wiresOp.getBodyBlock());

    // Connect condition to mux.cond
    wiresBuilder.create<calyx::AssignOp>(ifOp.getLoc(), muxOp.getCond(),
                                         condition);

    // Connect then value to mux.tru
    wiresBuilder.create<calyx::AssignOp>(ifOp.getLoc(), muxOp.getTru(),
                                         thenResult);

    // Connect else value to mux.fal
    wiresBuilder.create<calyx::AssignOp>(ifOp.getLoc(), muxOp.getFal(),
                                         elseResult);

    // Replace all uses of the scf.if result with the mux output
    ifResult.replaceAllUsesWith(muxOp.getOut());

    // Erase the original scf.if
    ifOp.erase();

    return success();
  }

  // Create constant 1 for write enable using deduplication
  auto constantOne = getOrCreateConstant(funcOp.getOperation(), 1);

  // Replace all uses of the scf.if result with the register output
  ifResult.replaceAllUsesWith(resultReg.getOut());

  // Replace yield operations with register assignments and group_done
  // operations Then region
  if (!thenRegion.empty()) {
    auto &thenBlock = thenRegion.front();
    for (auto &op : llvm::make_early_inc_range(thenBlock)) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(op)) {
        OpBuilder builder(&op);
        // Create assignment: reg.in = thenYieldValue
        builder.create<calyx::AssignOp>(op.getLoc(), resultReg.getIn(),
                                        thenYieldValue, ifOp.getCondition());
        // Create assignment: reg.write_en = 1
        builder.create<calyx::AssignOp>(op.getLoc(), resultReg.getWriteEn(),
                                        constantOne, ifOp.getCondition());
        // Create group_done
        builder.create<calyx::GroupDoneOp>(op.getLoc(), resultReg.getDone());
        yieldOp.erase();
      }
    }
  }

  // Else region
  if (!elseRegion.empty()) {
    auto &elseBlock = elseRegion.front();
    for (auto &op : llvm::make_early_inc_range(elseBlock)) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(op)) {
        OpBuilder builder(&op);
        // During scaffolding phase, we're still in a function, not a component
        // Create the NotLibOp at function level for now
        auto conditionType = condition.getType();
        auto invertedCondition = functionBuilder.create<calyx::NotLibOp>(
            ifOp.getLoc(), "inverted_cond_" + getOpUniqueName(ifOp),
            llvm::SmallVector<mlir::Type>{conditionType, conditionType});
        builder.create<calyx::AssignOp>(op.getLoc(), invertedCondition.getIn(),
                                        ifOp.getCondition());
        // Create assignment: reg.in = elseYieldValue
        builder.create<calyx::AssignOp>(op.getLoc(), resultReg.getIn(),
                                        elseYieldValue,
                                        invertedCondition.getOut());
        // Create assignment: reg.write_en = 1
        builder.create<calyx::AssignOp>(op.getLoc(), resultReg.getWriteEn(),
                                        constantOne,
                                        invertedCondition.getOut());
        // Create group_done
        builder.create<calyx::GroupDoneOp>(op.getLoc(), resultReg.getDone());
        yieldOp.erase();
      }
    }
  }

  // Now replace the scf.if with calyx.if
  OpBuilder builder(ifOp);
  auto calyxIfOp = builder.create<calyx::IfOp>(ifOp.getLoc(), condition,
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

  // Erase the original scf.if
  ifOp.erase();

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
  if (controlRegion.empty()) {
    return failure(); // Nothing to wrap
  }

  auto &controlBlock = controlRegion.front();
  if (controlBlock.empty()) {
    return failure(); // Nothing to wrap
  }

  // Find the parent component to access groups in the wires section
  auto componentOp = controlOp->getParentOfType<calyx::ComponentOp>();
  if (!componentOp) {
    return failure(); // No parent component found
  }

  // TODO: Before wrapping, check each enable signal and remove groups that only
  // have done For now, this optimization is disabled to avoid rewriter issues
  // Future work: Implement safe group removal that doesn't interfere with the
  // rewriter pattern

  // Re-collect control operations after potential removals
  SmallVector<Operation *> controlOps;
  bool hasStandaloneEnable = false;

  for (auto &op : controlBlock) {
    if (!isa<calyx::ControlOp>(op)) { // Don't count terminators
      controlOps.push_back(&op);
      if (isa<calyx::EnableOp>(op)) {
        hasStandaloneEnable = true;
      }
    }
  }

  // Need wrapping if:
  // 1. We have standalone enables, OR
  // 2. We have multiple control operations at the top level
  bool needsWrapping = hasStandaloneEnable || (controlOps.size() > 1);

  if (!needsWrapping) {
    return failure(); // Pattern doesn't apply
  }

  // Create a calyx.seq to wrap all operations
  auto loc = controlOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&controlBlock);

  auto seqOp = rewriter.create<calyx::SeqOp>(loc);
  auto &seqRegion = seqOp.getBodyRegion();
  auto &seqBlock = seqRegion.front();

  // Clone all control operations to the seq block
  // This handles both standalone enables and multiple control operations
  OpBuilder seqBuilder(&seqBlock, seqBlock.end());
  IRMapping mapper;
  for (auto *op : controlOps) {
    seqBuilder.clone(*op, mapper);
  }

  // Erase the original operations from the control block
  for (auto *op : controlOps) {
    rewriter.eraseOp(op);
  }

  return success();
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

  // Replace with calyx.repeat
  OpBuilder builder(forOp);
  auto calyxRepeatOp = builder.create<calyx::RepeatOp>(
      forOp.getLoc(), static_cast<uint32_t>(count));

  // Move the body region contents to the new RepeatOp
  auto &forRegion = forOp.getRegion();
  if (!forRegion.empty()) {
    auto &forBlock = forRegion.front();
    auto &repeatBlock = calyxRepeatOp.getBodyRegion().front();

    // Move all operations except the yield
    for (auto &op : llvm::make_early_inc_range(forBlock)) {
      if (isa<mlir::scf::YieldOp>(op)) {
        op.erase(); // Remove yield as Calyx repeat doesn't need it
        continue;
      }
      op.moveBefore(&repeatBlock, repeatBlock.end());
    }
  }

  // If the for loop had results, we need to handle them
  // For now, we'll create a placeholder - this might need more sophisticated
  // handling
  if (forResult) {
    // Find the wires operation and set insertion point before it
    auto &entryBlock = funcOp.front();
    auto wiresOps = entryBlock.getOps<calyx::WiresOp>();
    if (!wiresOps.empty()) {
      auto wiresOp = *wiresOps.begin();
      functionBuilder.setInsertionPoint(wiresOp);
    }

    // Create a temporary register to hold the result
    std::string regName = "for_result_" + getOpUniqueName(forOp);
    auto resultReg = functionBuilder.create<calyx::RegisterOp>(
        forOp.getLoc(), regName, forResult.getType());

    // Replace all uses of the for result with the register output
    forResult.replaceAllUsesWith(resultReg.getOut());
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
