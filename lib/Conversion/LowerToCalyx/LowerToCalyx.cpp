//===- LowerToCalyx.cpp - Unified Lowering Pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the unified LowerToCalyx pass that combines structural
// transformation phases with dialect-specific pattern-based lowering.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LowerToCalyx.h"
#include "LowerToCalyxUtil.h"
#include "circt/Dialect/Calyx/CalyxLoweringUtils.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.h"
#include "convertPattern.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cassert>

using namespace circt;
using namespace circt::calyx;
using namespace mlir;

// Forward declaration from ControlFlowToCalyx.cpp
namespace circt {
namespace lowertocalyx {}
} // namespace circt

namespace circt {
#define GEN_PASS_DEF_LOWERTOCALYX
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

namespace circt {
namespace lowertocalyx {

namespace {

class LowerToCalyxPass
    : public circt::impl::LowerToCalyxBase<LowerToCalyxPass> {
public:
  LowerToCalyxPass() = default;
  LowerToCalyxPass(std::string topLevelFunction) {
    topLevelFunctionOpt = std::move(topLevelFunction);
  }

  void runOnOperation() override;

private:
  /// Map from result value pointer to register name for lookup during pattern
  /// conversion
  DenseMap<uintptr_t, std::string> resultToRegisterName;

  /// Step 2: Apply SCF (Structured Control Flow) to Calyx conversion patterns
  LogicalResult applyControlFlowConversion(ModuleOp moduleOp);

  /// Step 4: Apply arithmetic patterns using the separated pattern classes
  LogicalResult applyArithPatterns(ModuleOp moduleOp);

  /// Step 0: Apply index type conversion patterns
  LogicalResult applyIndexConversionPatterns(ModuleOp moduleOp);

  /// Step 5: Apply memory patterns using the separated pattern classes
  LogicalResult applyMemoryPatterns(ModuleOp moduleOp);

  /// Step 6a: Apply function signature to component conversion
  LogicalResult applyFuncSignatureConversion(ModuleOp moduleOp);

  /// Step 6b: Apply return operation conversion
  LogicalResult applyReturnConversion(ModuleOp moduleOp);

  /// Step 7a: Apply empty group optimization before wrapping
  LogicalResult applyEmptyGroupOptimization(ModuleOp moduleOp);

  /// Step 7b: Apply control flow wrapping for calyx.seq/calyx.par
  LogicalResult applyControlFlowWrapping(ModuleOp moduleOp);

  /// Step 8b: Validate that all operations have been converted
  LogicalResult validateConversion(ModuleOp moduleOp);

  /// Step 2: Convert all basic blocks into Calyx groups.
  LogicalResult convertBlocksToGroups(ModuleOp moduleOp);

  /// Helper to check if a function should be processed
  bool shouldProcessFunction(mlir::func::FuncOp funcOp);
};

//===----------------------------------------------------------------------===//
// LowerToCalyxPass Implementation
//===----------------------------------------------------------------------===//

void LowerToCalyxPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  if (moduleOp.getOps<mlir::func::FuncOp>().empty()) {
    return; // Nothing to process
  }

  // Step 0: Normalize index types first so subsequent signature + control flow
  // conversion don't have to materialize i32<->index back-edges.
  if (failed(applyIndexConversionPatterns(moduleOp))) {
    signalPassFailure();
    return;
  }

  // Step 1: Convert function signatures to components with integrated
  // scaffolding. (Memrefs removed, indices already lowered to i32.)
  if (failed(applyFuncSignatureConversion(moduleOp))) {
    signalPassFailure();
    return;
  }

  // Step 2: Apply SCF (Structured Control Flow) conversion patterns
  if (failed(applyControlFlowConversion(moduleOp))) {
    signalPassFailure();
    return;
  }

  // Step 3: Convert arith operations (indices already converted)
  if (failed(applyArithPatterns(moduleOp))) {
    signalPassFailure();
    return;
  }

  // Step 3: Convert memref operations AFTER arithmetic (so index values are
  // converted)
  if (failed(applyMemoryPatterns(moduleOp))) {
    signalPassFailure();
    return;
  }

  // Step 4: Convert return operations (second part of function conversion)
  if (failed(applyReturnConversion(moduleOp))) {
    signalPassFailure();
    return;
  }

  if (failed(convertBlocksToGroups(moduleOp))) {
    signalPassFailure();
    return;
  }

  if (failed(applyControlFlowWrapping(moduleOp))) {
    signalPassFailure();
    return;
  }
}

LogicalResult LowerToCalyxPass::applyControlFlowConversion(ModuleOp moduleOp) {
  ConversionTarget target(getContext());
  target.addLegalDialect<calyx::CalyxDialect, arith::ArithDialect,
                         comb::CombDialect, hw::HWDialect>();

  // Allow memref operations during SCF conversion - they'll be converted later
  target.addLegalDialect<mlir::memref::MemRefDialect>();

  // Mark SCF and Affine operations as illegal to trigger conversion
  target.addIllegalOp<mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::WhileOp>();
  target.addIllegalOp<mlir::affine::AffineForOp>();

  RewritePatternSet patterns(&getContext());

  // Set up type converter with proper index type conversion and materialization
  TypeConverter typeConverter;

  // Convert index types to i32, pass other types through
  typeConverter.addConversion([](Type type) -> Type {
    if (type.isIndex()) {
      return IntegerType::get(type.getContext(), 32);
    }
    return type;
  });

  // Add source materialization (when converting from original to target type)
  typeConverter.addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    if (inputs.size() != 1) {
      return nullptr;
    }
    Value input = inputs[0];

    // Handle index to i32 conversion
    if (input.getType().isIndex() && isa<IntegerType>(resultType)) {
      auto intType = cast<IntegerType>(resultType);
      if (intType.getWidth() == 32) {
        return builder.create<arith::IndexCastOp>(loc, resultType, input)
            .getResult();
      }
    }

    return nullptr;
  });

  // Add target materialization (when converting from target to original type)
  typeConverter.addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    if (inputs.size() != 1) {
      return nullptr;
    }
    Value input = inputs[0];

    // Handle i32 to index conversion
    if (resultType.isIndex() && isa<IntegerType>(input.getType())) {
      auto intType = cast<IntegerType>(input.getType());
      if (intType.getWidth() == 32) {
        return builder.create<arith::IndexCastOp>(loc, resultType, input)
            .getResult();
      }
    }

    return nullptr;
  });

  // Add SCF and Affine to Calyx conversion patterns
  patterns.add<lowertocalyx::ScfIfToCalyxPattern>(typeConverter, &getContext());
  patterns.add<lowertocalyx::ScfForToCalyxPattern>(typeConverter,
                                                   &getContext());
  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

LogicalResult LowerToCalyxPass::convertBlocksToGroups(ModuleOp moduleOp) {
  // Get the first calyx::ComponentOp in the module
  calyx::ComponentOp componentOp =
      *(moduleOp.getOps<calyx::ComponentOp>().begin());
  auto wiresOp = componentOp.getWiresOp();
  auto controlOp = componentOp.getControlOp();
  OpBuilder wiresBuilder(&wiresOp.getBody().front(),
                         wiresOp.getBody().front().end());
  int groupCounter = 0;

  // Collect blocks to process to avoid iterator invalidation
  SmallVector<Block *> blocksToProcess;
  controlOp->walk<WalkOrder::PreOrder>([&](Block *block) {
    if (!block->empty()) {
      blocksToProcess.push_back(block);
    }
  });

  // Process each block using basic block walk with operation grouping
  for (Block *block : blocksToProcess) {
    if (block->empty()) {
      continue; // Skip empty blocks
    }

    // Construct list of operation groups during block walk
    SmallVector<SmallVector<Operation *>> operationGroups;
    SmallVector<Operation *> currentGroup;

    // Walk through operations in the block to collect groups
    for (auto &op : *block) {
      if (op.hasTrait<calyx::ControlLike>()) {
        // Encountered a control operation - this splits the block
        if (!currentGroup.empty()) {
          operationGroups.push_back(std::move(currentGroup));
          currentGroup.clear();
        }
        // Control operations are not moved to groups - they stay in control
        continue;
      }
      // Add non-control operation to current group
      currentGroup.push_back(&op);
    }

    // Add the last group if it contains operations
    if (!currentGroup.empty()) {
      operationGroups.push_back(std::move(currentGroup));
    }

    // Now replace operation groups with enable operations
    // We need to process this carefully to maintain insertion order
    SmallVector<Operation *> operationsToRemove;
    OpBuilder blockBuilder(block, block->begin());

    for (auto &opGroup : operationGroups) {
      if (opGroup.empty()) {
        continue;
      }

      // Create the group
      std::string groupName = "bb" + std::to_string(groupCounter++);
      auto groupOp =
          wiresBuilder.create<calyx::GroupOp>(componentOp.getLoc(), groupName);
      Block *groupBodyBlock = groupOp.getBodyBlock();

      bool haveDoneOp = false;

      // Find the position of the first operation in this group to place the
      // enable
      Operation *firstOpInGroup = opGroup[0];
      blockBuilder.setInsertionPoint(firstOpInGroup);

      // Create enable operation at the position of the first operation in the
      // group
      blockBuilder.create<calyx::EnableOp>(componentOp.getLoc(),
                                           groupOp.getSymName());

      // Move all operations in this group to the group body
      for (Operation *op : opGroup) {
        operationsToRemove.push_back(op);
        op->moveBefore(groupBodyBlock, groupBodyBlock->end());
        if (isa<calyx::GroupDoneOp>(*op)) {
          haveDoneOp = true;
        }
      }

      // Handle done signal for the group
      if (!haveDoneOp) {
        // Look for the last done signal in the group
        Value lastDoneSignal = nullptr;

        // Collect all unique done signals, then use the last one
        llvm::SmallPtrSet<Value, 4> uniqueDoneSignals;

        // Iterate through the operations in the group to find done signals
        for (auto &groupBodyOp : groupBodyBlock->getOperations()) {
          // Skip the GroupDoneOp itself
          if (isa<calyx::GroupDoneOp>(groupBodyOp)) {
            continue;
          }
          // Look for operations that produce done signals
          if (auto assign = dyn_cast<calyx::AssignOp>(&groupBodyOp)) {
            Value doneSignal =
                resolveDoneSignalForValue(assign.getDest(), componentOp);
            if (doneSignal) {
              uniqueDoneSignals.insert(doneSignal);
              lastDoneSignal = doneSignal; // Keep updating to get the last one
            }
          } else {
            for (Value result : groupBodyOp.getResults()) {
              Value doneSignal = resolveDoneSignalForValue(result, componentOp);
              if (doneSignal) {
                uniqueDoneSignals.insert(doneSignal);
                lastDoneSignal =
                    doneSignal; // Keep updating to get the last one
              }
            }
          }
        }

        if (lastDoneSignal) {
          // Use the last found done signal for the group done - create only ONE
          // GroupDoneOp
          OpBuilder groupBuilder(groupBodyBlock, groupBodyBlock->end());
          groupBuilder.create<calyx::GroupDoneOp>(componentOp.getLoc(),
                                                  lastDoneSignal);
        } else {
          // No done signals found - mark this group as combinational
          // In Calyx, combinational groups don't have done signals
          // We can set the "comb" attribute on the group
          groupOp->setAttr("comb",
                           mlir::UnitAttr::get(componentOp.getContext()));
        }
      }
    }
  }

  return success();
}

LogicalResult LowerToCalyxPass::applyArithPatterns(ModuleOp moduleOp) {
  ConversionTarget target(getContext());
  target.addLegalDialect<calyx::CalyxDialect, hw::HWDialect>();

  target.addIllegalDialect<arith::ArithDialect>();
  RewritePatternSet patterns(&getContext());

  // Set up type converter with proper index type conversion
  TypeConverter typeConverter;

  // Convert index types to i32, pass other types through
  typeConverter.addConversion([](Type type) -> Type {
    if (isa<IndexType>(type)) {
      return IntegerType::get(type.getContext(), 32);
    }
    return type;
  });

  // Mirror the source/target materializations used earlier so that any
  // intermediate casts introduced (or elided) during IndexCast lowering are
  // properly legalized instead of leaving an unrealized_conversion_cast.
  typeConverter.addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    if (inputs.size() != 1)
      return nullptr;
    Value input = inputs[0];
    // index -> i32 (source side when original IR had index)
    // Handle index to i32 conversion
    if (input.getType().isIndex() && isa<IntegerType>(resultType)) {
      auto intType = cast<IntegerType>(resultType);
      if (intType.getWidth() == 32) {
        return builder.create<arith::IndexCastOp>(loc, resultType, input)
            .getResult();
      }
    }
    return nullptr;
  });

  typeConverter.addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    if (inputs.size() != 1) {
      return nullptr;
    }
    Value input = inputs[0];

    // Handle i32 to index conversion
    if (resultType.isIndex() && isa<IntegerType>(input.getType())) {
      auto intType = cast<IntegerType>(input.getType());
      if (intType.getWidth() == 32) {
        return builder.create<arith::IndexCastOp>(loc, resultType, input)
            .getResult();
      }
    }

    return nullptr;
  });
  // Add arithmetic patterns directly
  patterns.add<lowertocalyx::ArithAddIToCalyxPattern,
               lowertocalyx::ArithSubIToCalyxPattern,
               lowertocalyx::ArithMulIToCalyxPattern,
               lowertocalyx::ArithAndIToCalyxPattern,
               lowertocalyx::ArithOrIToCalyxPattern,
               lowertocalyx::ArithXOrIToCalyxPattern,
               lowertocalyx::ArithExtSIToCalyxPattern,
               lowertocalyx::ArithTruncIToCalyxPattern,
               lowertocalyx::ArithCmpIToCalyxPattern,
               lowertocalyx::ArithConstantToCalyxPattern,
               lowertocalyx::ArithIndexCastToCalyxPattern,
               lowertocalyx::ArithShRUIToCalyxPattern,
               lowertocalyx::ArithShRSIToCalyxPattern,
               lowertocalyx::ArithShLIToCalyxPattern,
               lowertocalyx::ArithSelectToCalyxPattern>(typeConverter,
                                                        &getContext());
  moduleOp.dump();
  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

LogicalResult LowerToCalyxPass::applyMemoryPatterns(ModuleOp moduleOp) {
  // Apply memory patterns
  ConversionTarget target(getContext());
  target
      .addLegalDialect<calyx::CalyxDialect, comb::CombDialect, hw::HWDialect>();

  // Mark alloca/malloc-based memref operations as illegal
  target.addIllegalOp<mlir::memref::AllocOp, mlir::memref::AllocaOp>();

  // Mark memref operations as illegal when in component context (after function
  // conversion)
  target.addDynamicallyLegalOp<mlir::memref::LoadOp>(
      [](mlir::memref::LoadOp loadOp) {
        // Check if we're in a component context (after function conversion)
        if (loadOp->getParentOfType<calyx::ComponentOp>()) {
          return false; // Illegal if in component
        }
        if (loadOp->getParentOfType<mlir::func::FuncOp>()) {
          return true; // Legal if still in function
        }
        return true;
      });

  target.addDynamicallyLegalOp<mlir::memref::StoreOp>(
      [](mlir::memref::StoreOp storeOp) {
        // Check if we're in a component context (after function conversion)
        if (storeOp->getParentOfType<calyx::ComponentOp>()) {
          return false; // Illegal if in component
        }
        if (storeOp->getParentOfType<mlir::func::FuncOp>()) {
          return true; // Legal if still in function
        }
        return true;
      });

  RewritePatternSet patterns(&getContext());

  // Set up type converter with proper index type conversion
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) -> Type {
    if (type.isIndex()) {
      return IntegerType::get(type.getContext(), 32);
    }
    return type;
  });

  // Add complete memref conversion patterns
  patterns.add<lowertocalyx::MemrefLoadToCalyxPattern,
               lowertocalyx::MemrefStoreToCalyxPattern,
               lowertocalyx::MemrefAllocaToCalyxPattern>(typeConverter,
                                                         &getContext());

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

LogicalResult
LowerToCalyxPass::applyIndexConversionPatterns(ModuleOp moduleOp) {
  ConversionTarget target(getContext());
  target.addLegalDialect<calyx::CalyxDialect, arith::ArithDialect,
                         comb::CombDialect, hw::HWDialect>();

  // We want to convert functions that have index types
  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp funcOp) {
    auto funcType = funcOp.getFunctionType();
    // Legal if no index types in signature or arguments
    for (Type inputType : funcType.getInputs()) {
      if (inputType.isIndex())
        return false;
    }
    for (Type resultType : funcType.getResults()) {
      if (resultType.isIndex())
        return false;
    }
    for (Value arg : funcOp.getArguments()) {
      if (arg.getType().isIndex())
        return false;
    }
    return true;
  });

  RewritePatternSet patterns(&getContext());

  // Set up type converter for index types
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) -> Type {
    if (type.isIndex()) {
      return IntegerType::get(type.getContext(), 32);
    }
    return type;
  });

  // Add index conversion patterns
  patterns.add<lowertocalyx::FuncOpIndexConversionPattern>(typeConverter,
                                                           &getContext());

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

LogicalResult
LowerToCalyxPass::applyFuncSignatureConversion(ModuleOp moduleOp) {
  ConversionTarget target(getContext());
  target.addLegalDialect<calyx::CalyxDialect>();
  target
      .addLegalDialect<arith::ArithDialect, comb::CombDialect, hw::HWDialect>();

  // Allow memref operations during function conversion - they'll be converted
  // later
  target.addLegalDialect<mlir::memref::MemRefDialect>();

  // Mark function operations as illegal
  target.addIllegalOp<mlir::func::FuncOp>();

  // Set up type converter with complete type conversion including memref
  // removal
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) -> Type {
    if (dyn_cast<IndexType>(type)) {
      return IntegerType::get(type.getContext(), 32);
    }
    return type;
  });
  // memref args are removed from signature and become internal memory
  typeConverter.addConversion([](MemRefType) -> Type { return nullptr; });

  RewritePatternSet patterns(&getContext());
  // Add function signature conversion pattern only
  patterns.add<lowertocalyx::FuncFuncToCalyxPattern>(typeConverter,
                                                     &getContext());

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    return failure();
  }

  // Set toplevel attribute after function conversion
  auto componentOps = moduleOp.getOps<calyx::ComponentOp>();
  if (std::distance(componentOps.begin(), componentOps.end()) == 1) {
    calyx::ComponentOp componentOp = *componentOps.begin();
    componentOp->setAttr("toplevel", UnitAttr::get(moduleOp.getContext()));
  } else {
    Operation *topLevelComponent =
        SymbolTable::lookupSymbolIn(moduleOp, topLevelFunctionOpt);
    if (!topLevelComponent) {
      moduleOp.emitError("Top-level component not found: ")
          << topLevelFunctionOpt;
      return failure();
    }

    topLevelComponent->setAttr("toplevel",
                               UnitAttr::get(moduleOp.getContext()));
  }

  return success();
}

LogicalResult LowerToCalyxPass::applyReturnConversion(ModuleOp moduleOp) {
  ConversionTarget target(getContext());
  target.addLegalDialect<calyx::CalyxDialect>();
  target
      .addLegalDialect<arith::ArithDialect, comb::CombDialect, hw::HWDialect>();

  // Allow memref operations during return conversion - they should be converted
  // by now
  target.addLegalDialect<mlir::memref::MemRefDialect>();

  // Mark func.return as illegal so it gets converted
  target.addIllegalOp<mlir::func::ReturnOp>();

  // Set up type converter with complete type conversion including memref
  // removal
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) -> Type {
    if (type.isIndex()) {
      return IntegerType::get(type.getContext(), 32);
    }
    return type;
  });
  // memref args are removed from signature and become internal memory
  typeConverter.addConversion([](MemRefType) -> Type { return nullptr; });

  RewritePatternSet patterns(&getContext());
  // Add return conversion pattern only
  patterns.add<lowertocalyx::FuncReturnToCalyxPattern>(typeConverter,
                                                       &getContext());

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

LogicalResult LowerToCalyxPass::validateConversion(ModuleOp moduleOp) {
  // Walk through the module and ensure no illegal operations remain
  auto result = success();
  moduleOp.walk([&](Operation *op) {
    // Check for operations that should have been converted
    if (isa<mlir::arith::AddIOp, mlir::arith::SubIOp, mlir::arith::MulIOp,
            mlir::arith::CmpIOp, mlir::arith::ConstantOp, mlir::scf::IfOp,
            mlir::scf::ForOp, mlir::scf::WhileOp, mlir::func::FuncOp,
            mlir::func::ReturnOp, mlir::memref::AllocOp, mlir::memref::AllocaOp,
            mlir::memref::LoadOp, mlir::memref::StoreOp,
            mlir::memref::GetGlobalOp>(op)) {
      op->emitError(
          "Operation should have been converted by LowerToCalyx pass");
      result = failure();
    }
  });
  return result;
}

bool LowerToCalyxPass::shouldProcessFunction(mlir::func::FuncOp funcOp) {
  // Skip external functions
  if (funcOp.isExternal())
    return false;

  // If top-level function is specified, only process that one
  if (!topLevelFunctionOpt.empty() && funcOp.getName() != topLevelFunctionOpt)
    return false;

  return true;
}

LogicalResult LowerToCalyxPass::applyEmptyGroupOptimization(ModuleOp moduleOp) {
  ConversionTarget target(getContext());
  target.addLegalDialect<calyx::CalyxDialect, arith::ArithDialect,
                         comb::CombDialect, hw::HWDialect>();

  // Helper function to check if a component has empty groups
  auto hasEmptyGroups = [](calyx::ComponentOp componentOp) -> bool {
    auto wiresOp = componentOp.getWiresOp();
    if (!wiresOp)
      return false;

    for (auto groupOp : wiresOp.getOps<calyx::GroupOp>()) {
      auto *groupBlock = groupOp.getBodyBlock();
      if (!groupBlock)
        continue;

      size_t nonDoneOpsCount = 0;
      bool hasDone = false;

      for (auto &op : *groupBlock) {
        if (isa<calyx::GroupDoneOp>(op)) {
          hasDone = true;
        } else {
          nonDoneOpsCount++;
        }
      }

      if (hasDone && (nonDoneOpsCount == 0)) {
        return true; // Has empty groups
      }
    }
    return false;
  };

  // We want to convert components that have empty groups
  target.addDynamicallyLegalOp<calyx::ComponentOp>(
      [hasEmptyGroups](calyx::ComponentOp componentOp) {
        return !hasEmptyGroups(componentOp); // Legal if no empty groups
      });

  RewritePatternSet patterns(&getContext());

  // Set up type converter
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) -> Type { return type; });

  // Add only the empty group optimization pattern
  patterns.add<lowertocalyx::EmptyGroupOptimizationPattern>(typeConverter,
                                                            &getContext());

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

LogicalResult LowerToCalyxPass::applyControlFlowWrapping(ModuleOp moduleOp) {
  ConversionTarget target(getContext());
  target.addLegalDialect<calyx::CalyxDialect, arith::ArithDialect,
                         comb::CombDialect, hw::HWDialect>();

  // We want to convert control operations that need wrapping
  target.addDynamicallyLegalOp<calyx::ControlOp>(
      [](calyx::ControlOp controlOp) {
        auto &controlRegion = controlOp.getBodyRegion();
        if (controlRegion.empty()) {
          return true; // Legal if empty
        }

        auto &controlBlock = controlRegion.front();
        if (controlBlock.empty()) {
          return true; // Legal if empty
        }

        // Check if we need wrapping - look for problematic patterns
        size_t numOperations = 0;
        bool hasStandaloneEnable = false;

        for (auto &op : controlBlock) {
          if (!isa<calyx::ControlOp>(op)) { // Don't count the terminator
            numOperations++;
            if (isa<calyx::EnableOp>(op)) {
              hasStandaloneEnable = true;
            }
          }
        }

        // Legal if we don't need wrapping
        bool needsWrapping = (numOperations > 1 && hasStandaloneEnable) ||
                             (numOperations == 1 && hasStandaloneEnable);
        return !needsWrapping;
      });

  // Control flow wrapping operates only on control operations
  // Empty group optimization is now handled in a separate pass

  RewritePatternSet patterns(&getContext());

  // Set up type converter
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) -> Type { return type; });

  // Add control flow wrapping patterns
  // Note: Empty group optimization disabled due to rewriter conflicts
  patterns.add<lowertocalyx::ControlFlowWrappingPattern>(typeConverter,
                                                         &getContext());

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>>
createLowerToCalyxPass(const std::string &topLevelFunction) {
  return std::make_unique<LowerToCalyxPass>(topLevelFunction);
}

} // namespace lowertocalyx
} // namespace circt

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace circt {
std::unique_ptr<OperationPass<ModuleOp>>
createLowerToCalyxPass(std::string topLevelFunction) {
  return lowertocalyx::createLowerToCalyxPass(std::move(topLevelFunction));
}
} // namespace circt