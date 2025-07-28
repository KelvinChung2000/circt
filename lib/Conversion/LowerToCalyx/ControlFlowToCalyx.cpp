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

/// Transform a single SCF If operation to Calyx hardware constructs
/// This function creates registers, not ops, and assigns directly in the
/// transformation
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
  auto calyxRegOp = functionBuilder.create<calyx::RegisterOp>(
      ifOp.getLoc(), regName, ifResult.getType());

  Value regIn = calyxRegOp.getResult(0);      // reg.in
  Value regWriteEn = calyxRegOp.getResult(1); // reg.write_en
  Value regOut = calyxRegOp.getResult(4);     // reg.out
  Value regDone = calyxRegOp.getResult(5);    // reg.done

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

  // Create constant 1 for write enable using deduplication
  auto constantOne = getOrCreateConstant(funcOp.getOperation(), functionBuilder, 1);

  // Replace all uses of the scf.if result with the register output
  ifResult.replaceAllUsesWith(regOut);

  // Replace yield operations with register assignments and group_done
  // operations Then region
  if (!thenRegion.empty()) {
    auto &thenBlock = thenRegion.front();
    for (auto &op : llvm::make_early_inc_range(thenBlock)) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(op)) {
        OpBuilder builder(&op);
        // Create assignment: reg.in = thenYieldValue
        builder.create<calyx::AssignOp>(op.getLoc(), regIn, thenYieldValue,
                                        ifOp.getCondition());
        // Create assignment: reg.write_en = 1
        builder.create<calyx::AssignOp>(op.getLoc(), regWriteEn,
                                        constantOne,
                                        ifOp.getCondition());
        // Create group_done
        builder.create<calyx::GroupDoneOp>(op.getLoc(), regDone);
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
        builder.create<calyx::AssignOp>(op.getLoc(), regIn, elseYieldValue,
                                        invertedCondition.getOut());
        // Create assignment: reg.write_en = 1
        builder.create<calyx::AssignOp>(op.getLoc(), regWriteEn,
                                        constantOne,
                                        invertedCondition.getOut());
        // Create group_done
        builder.create<calyx::GroupDoneOp>(op.getLoc(), regDone);
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

template <>
LogicalResult ScfToCalyxPattern<mlir::scf::ForOp>::matchAndRewrite(
    mlir::scf::ForOp op, mlir::scf::ForOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Check if this is a constant-bound for loop that can be converted to repeat
  if (auto lbConstant =
          op.getLowerBound().getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto ubConstant =
            op.getUpperBound().getDefiningOp<mlir::arith::ConstantOp>()) {
      if (auto stepConstant =
              op.getStep().getDefiningOp<mlir::arith::ConstantOp>()) {
        // Extract constant values
        auto lbIntAttr =
            mlir::dyn_cast<mlir::IntegerAttr>(lbConstant.getValue());
        auto ubIntAttr =
            mlir::dyn_cast<mlir::IntegerAttr>(ubConstant.getValue());
        auto stepIntAttr =
            mlir::dyn_cast<mlir::IntegerAttr>(stepConstant.getValue());

        if (!lbIntAttr || !ubIntAttr || !stepIntAttr) {
          return rewriter.notifyMatchFailure(
              op, "non-integer constant bounds in for loop");
        }

        auto lbValue = lbIntAttr.getInt();
        auto ubValue = ubIntAttr.getInt();
        auto stepValue = stepIntAttr.getInt();

        // Calculate iteration count
        if (stepValue > 0 && ubValue > lbValue) {
          int64_t count = (ubValue - lbValue + stepValue - 1) / stepValue;

          // Create a Calyx RepeatOp
          auto calyxRepeatOp = rewriter.create<calyx::RepeatOp>(
              op.getLoc(), static_cast<uint32_t>(count));

          // Move the body region contents to the new RepeatOp
          rewriter.inlineRegionBefore(op.getRegion(),
                                      calyxRepeatOp.getBodyRegion(),
                                      calyxRepeatOp.getBodyRegion().begin());

          // Replace the original ForOp with the new Calyx RepeatOp
          rewriter.eraseOp(op);

          return success();
        }
      }
    }
  }

  // For non-constant bounds, mark as not implemented
  return rewriter.notifyMatchFailure(
      op, "dynamic for loops not implemented - only constant-bound for loops "
          "can be converted to calyx.repeat");
}

template <>
LogicalResult ScfToCalyxPattern<mlir::scf::WhileOp>::matchAndRewrite(
    mlir::scf::WhileOp op, mlir::scf::WhileOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Get the condition from the before region
  Block *beforeBlock =
      op.getBefore().getBlocks().empty() ? nullptr : &op.getBefore().front();
  if (!beforeBlock) {
    return rewriter.notifyMatchFailure(op, "while op has no before region");
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
    return rewriter.notifyMatchFailure(op,
                                       "could not find condition in while op");
  }

  // Create a new Calyx WhileOp with the condition
  auto calyxWhileOp =
      rewriter.create<calyx::WhileOp>(op.getLoc(), condition, nullptr);

  // Move the after region (body) contents to the new WhileOp
  if (!op.getAfter().empty()) {
    rewriter.inlineRegionBefore(op.getAfter(), calyxWhileOp.getBodyRegion(),
                                calyxWhileOp.getBodyRegion().begin());
  }

  // Replace the original WhileOp with the new Calyx WhileOp
  rewriter.eraseOp(op);

  return success();
}

// Helper function to check if a group only contains done signals
static bool groupOnlyHasDone(calyx::GroupOp groupOp) {
  auto *groupBlock = groupOp.getBodyBlock();
  if (!groupBlock) return false;
  
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
static calyx::GroupOp findReferencedGroup(calyx::EnableOp enableOp, calyx::ComponentOp componentOp) {
  StringRef groupName = enableOp.getGroupName();
  
  // Search for the group in the component's wires section
  auto wiresOp = componentOp.getWiresOp();
  if (!wiresOp) return nullptr;
  
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

  // TODO: Before wrapping, check each enable signal and remove groups that only have done
  // For now, this optimization is disabled to avoid rewriter issues
  // Future work: Implement safe group removal that doesn't interfere with the rewriter pattern

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

template <>
LogicalResult ScfToCalyxPattern<mlir::scf::YieldOp>::matchAndRewrite(
    mlir::scf::YieldOp op, mlir::scf::YieldOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // For yield operations, we typically just erase them since Calyx control flow
  // operations handle termination automatically. The operands (if any) are
  // handled by the parent control flow operation.

  // In some cases, we might need to handle yield operands differently based on
  // the parent operation context, but for now, we'll simply erase the yield.
  rewriter.eraseOp(op);

  return success();
}

template <>
LogicalResult ScfToCalyxPattern<mlir::scf::ExecuteRegionOp>::matchAndRewrite(
    mlir::scf::ExecuteRegionOp op, mlir::scf::ExecuteRegionOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // ExecuteRegionOp conversion not implemented
  return rewriter.notifyMatchFailure(
      op, "scf.execute_region conversion not implemented");
}

template <>
LogicalResult ScfToCalyxPattern<mlir::scf::ForallOp>::matchAndRewrite(
    mlir::scf::ForallOp op, mlir::scf::ForallOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // ForallOp (parallel for) conversion not implemented
  return rewriter.notifyMatchFailure(op,
                                     "scf.forall conversion not implemented");
}

template <>
LogicalResult ScfToCalyxPattern<mlir::scf::IndexSwitchOp>::matchAndRewrite(
    mlir::scf::IndexSwitchOp op, mlir::scf::IndexSwitchOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // IndexSwitchOp conversion not implemented
  return rewriter.notifyMatchFailure(
      op, "scf.index_switch conversion not implemented");
}

template <>
LogicalResult ScfToCalyxPattern<mlir::scf::ParallelOp>::matchAndRewrite(
    mlir::scf::ParallelOp op, mlir::scf::ParallelOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // ParallelOp conversion not implemented
  return rewriter.notifyMatchFailure(op,
                                     "scf.parallel conversion not implemented");
}

template <>
LogicalResult ScfToCalyxPattern<mlir::scf::ConditionOp>::matchAndRewrite(
    mlir::scf::ConditionOp op, mlir::scf::ConditionOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // The condition operation is handled by the parent while operation.
  // We typically erase the condition operation since the condition value
  // is extracted and used by the while operation conversion.
  rewriter.eraseOp(op);

  return success();
}

} // namespace lowertocalyx
} // namespace circt
