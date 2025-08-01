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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cassert>

using namespace circt;
using namespace circt::calyx;
using namespace mlir;

// Forward declaration from ControlFlowToCalyx.cpp
namespace circt {
namespace lowertocalyx {
LogicalResult transformScfIfToCalyx(mlir::scf::IfOp ifOp,
                                    OpBuilder &wiresBuilder,
                                    OpBuilder &functionBuilder);
}
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
  /// Step 1: Set up the basic Calyx component structure within each function.
  LogicalResult scaffoldCalyxStructure(mlir::func::FuncOp funcOp);

  /// Step 2: Convert all basic blocks into Calyx groups.
  LogicalResult convertBlocksToGroups(ModuleOp moduleOp);

  /// Step 4: Apply arithmetic patterns using the separated pattern classes
  LogicalResult applyArithPatterns(ModuleOp moduleOp);

  /// Step 0: Apply index type conversion patterns
  LogicalResult applyIndexConversionPatterns(ModuleOp moduleOp);

  /// Step 5: Apply memory patterns using the separated pattern classes
  LogicalResult applyMemoryPatterns(ModuleOp moduleOp);

  /// Step 6: Apply complete function to component conversion (final step)
  LogicalResult applyCompleteFunctionConversion(ModuleOp moduleOp);

  /// Step 7a: Apply empty group optimization before wrapping
  LogicalResult applyEmptyGroupOptimization(ModuleOp moduleOp);

  /// Step 7b: Apply control flow wrapping for calyx.seq/calyx.par
  LogicalResult applyControlFlowWrapping(ModuleOp moduleOp);

  /// Step 8b: Validate that all operations have been converted
  LogicalResult validateConversion(ModuleOp moduleOp);

  /// Helper to check if a function should be processed
  bool shouldProcessFunction(mlir::func::FuncOp funcOp);
};

//===----------------------------------------------------------------------===//
// LowerToCalyxPass Implementation
//===----------------------------------------------------------------------===//

void LowerToCalyxPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  // Step 0: Index conversion is now integrated into function conversion
  // to handle type coherence properly

  // Step 1: Add wires section and wrap function body in control op
  // (scaffolding)
  // SmallVector<mlir::func::FuncOp> functionsToProcess;
  // for (auto funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
  //   if (shouldProcessFunction(funcOp)) {
  //     functionsToProcess.push_back(funcOp);
  //   }
  // }

  if (moduleOp.getOps<mlir::func::FuncOp>().empty()) {
    return; // Nothing to process
  }

  // Process each function with scaffolding
  for (auto funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
    if (failed(scaffoldCalyxStructure(funcOp))) {
      signalPassFailure();
      return;
    }
  }

  // Step 5: Convert arith operations to equivalent std ops (patterns)
  if (failed(applyArithPatterns(moduleOp))) {
    signalPassFailure();
    return;
  }

  // Step 4: Convert memref operations BEFORE arithmetic (to establish proper
  // value dependencies)
  if (failed(applyMemoryPatterns(moduleOp))) {
    signalPassFailure();
    return;
  }

  // Step 3: Convert function to component (patterns) - After scaffolding
  if (failed(applyCompleteFunctionConversion(moduleOp))) {
    signalPassFailure();
    return;
  }

  if (failed(convertBlocksToGroups(moduleOp))) {
    signalPassFailure();
    return;
  }

  // // Step 6: Apply empty group optimization first, then control flow wrapping
  // if (failed(applyEmptyGroupOptimization(moduleOp))) {
  //   signalPassFailure();
  //   return;
  // }

  if (failed(applyControlFlowWrapping(moduleOp))) {
    signalPassFailure();
    return;
  }

  // Step 8b: Validation
  // if (failed(validateConversion(moduleOp))) {
  //   signalPassFailure();
  //   return;
  // }
}

LogicalResult
LowerToCalyxPass::scaffoldCalyxStructure(mlir::func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());

  // Get function information before conversion
  std::string componentName = funcOp.getName().str();
  auto loc = funcOp.getLoc();
  // Save the original blocks before creating new structure
  auto &funcBlocks = funcOp.getBlocks();
  llvm::SmallVector<Block *> originalBlocks;
  for (auto &block : funcBlocks) {
    originalBlocks.push_back(&block);
  }

  // Save the original entry block (with arguments) and clear its operations
  Block *entryBlock = &funcOp.getBody().front();
  SmallVector<std::unique_ptr<Block>> tempBlocks;

  // Move the entry block operations to temp storage
  tempBlocks.push_back(std::make_unique<Block>());
  tempBlocks.back()->getOperations().splice(tempBlocks.back()->begin(),
                                            entryBlock->getOperations());

  // Move any additional blocks to temp storage
  for (auto it = std::next(originalBlocks.begin()); it != originalBlocks.end();
       ++it) {
    Block *block = *it;
    tempBlocks.push_back(std::unique_ptr<Block>(block));
    block->getParent()->getBlocks().remove(block);
  }

  // Set insertion point to the cleared entry block (preserves arguments)
  builder.setInsertionPointToStart(entryBlock);

  // Create the `calyx.wires` and `calyx.control` ops in the function
  // Note: WiresOp and ControlOp automatically create blocks via their custom
  // builders
  auto wiresOp = builder.create<calyx::WiresOp>(loc);
  auto controlOp = builder.create<calyx::ControlOp>(loc);

  // Use the automatically created blocks (do not create additional blocks)
  OpBuilder wiresBuilder(&wiresOp.getBody().front(),
                         wiresOp.getBody().front().begin());

  // Use the automatically created block in the control region
  auto &controlRegion = controlOp.getBodyRegion();
  Block *controlBlock = &controlRegion.front();

  // Move all operations from temp blocks into the single control block
  OpBuilder controlBuilder(controlBlock, controlBlock->begin());
  for (auto &blockPtr : tempBlocks) {
    // Move all operations from each temp block to the single control block
    controlBlock->getOperations().splice(controlBlock->end(),
                                         blockPtr->getOperations());
  }

  // Transform SCF operations to Calyx hardware constructs
  // This is the primary place for SCF conversion - creates all necessary
  // registers and group_done ops
  for (auto &block : controlRegion.getBlocks()) {
    // Look for SCF If operations
    block.walk([&](mlir::scf::IfOp ifOp) {
      if (failed(transformScfIfToCalyx(ifOp, wiresBuilder, builder))) {
        // Handle error if needed
      }
    });

    // Look for SCF For operations
    block.walk([&](mlir::scf::ForOp forOp) {
      if (failed(transformScfForToCalyx(forOp, wiresBuilder, builder))) {
        // Handle error if needed
      }
    });

    // Look for SCF While operations
    block.walk([&](mlir::scf::WhileOp whileOp) {
      if (failed(transformScfWhileToCalyx(whileOp, wiresBuilder, builder))) {
        // Handle error if needed
      }
    });
  }

  return success();
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

  // Process each block - including blocks within calyx.if operations
  for (Block *block : blocksToProcess) {
    // Process ALL blocks, including those within calyx.if operations
    // The block-to-groups phase is responsible for creating all groups

    // Collect non-control operations that should be moved to groups
    SmallVector<Operation *> opsToMove;
    SmallVector<Operation *> groupDoneOps;
    if (block->empty()) {
      continue; // Skip empty blocks
    }

    bool haveDoneOp = false;
    std::string groupName = "bb" + std::to_string(groupCounter++);
    auto groupOp =
        wiresBuilder.create<calyx::GroupOp>(componentOp.getLoc(), groupName);

    Block *groupBodyBlock = groupOp.getBodyBlock();
    for (auto &op : llvm::make_early_inc_range(*block)) {
      if (op.hasTrait<calyx::ControlLike>()) {
        continue;
      }
      op.moveBefore(groupBodyBlock, groupBodyBlock->end());
      if (isa<calyx::GroupDoneOp>(op)) {
        haveDoneOp = true;
      }
    }
    if (!haveDoneOp) {
      // Instead of creating a constant done signal, look for the last done
      // signal in the group
      Value lastDoneSignal = nullptr;

      // Iterate through the operations in the group to find done signals
      for (auto &op : groupBodyBlock->getOperations()) {
        // Skip the GroupDoneOp itself
        if (isa<calyx::GroupDoneOp>(op)) {
          continue;
        }
        // Look for operations that produce done signals
        if (auto assign = dyn_cast<calyx::AssignOp>(&op)) {
          lastDoneSignal =
              resolveDoneSignalForValue(assign.getDest(), componentOp);
        } else {
          for (Value result : op.getResults()) {
            lastDoneSignal = resolveDoneSignalForValue(result, componentOp);
          }
        }
      }

      if (lastDoneSignal) {
        // Use the last found done signal for the group done
        OpBuilder groupBuilder(groupBodyBlock, groupBodyBlock->end());
        groupBuilder.create<calyx::GroupDoneOp>(componentOp.getLoc(),
                                                lastDoneSignal);
      } else {
        // No done signals found - mark this group as combinational
        // In Calyx, combinational groups don't have done signals
        // We can set the "comb" attribute on the group
        groupOp->setAttr("comb", mlir::UnitAttr::get(componentOp.getContext()));
      }
    }
    OpBuilder blockBuilder(block, block->begin());
    blockBuilder.create<calyx::EnableOp>(componentOp.getLoc(),
                                         groupOp.getSymName());
  }

  return success();
}

LogicalResult LowerToCalyxPass::applyArithPatterns(ModuleOp moduleOp) {
  ConversionTarget target(getContext());
  target.addLegalDialect<calyx::CalyxDialect, hw::HWDialect>();

  target.addIllegalDialect<arith::ArithDialect>();
  RewritePatternSet patterns(&getContext());

  // Set up type converter
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });

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
               lowertocalyx::ArithSelectToCalyxPattern,
               lowertocalyx::ArithIndexCastToCalyxPattern>(typeConverter,
                                                           &getContext());

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

  // Set up type converter
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });

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
LowerToCalyxPass::applyCompleteFunctionConversion(ModuleOp moduleOp) {
  ConversionTarget target(getContext());
  target.addLegalDialect<calyx::CalyxDialect>();
  target
      .addLegalDialect<arith::ArithDialect, comb::CombDialect, hw::HWDialect>();

  // Allow memref operations during function conversion - they'll be converted
  // later
  target.addLegalDialect<mlir::memref::MemRefDialect>();

  // Mark function operations as illegal
  target.addIllegalOp<mlir::func::FuncOp>();
  // Mark func.return as illegal so it gets converted
  // target.addIllegalOp<mlir::func::ReturnOp>();

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
  // Add complete function conversion pattern
  patterns.add<lowertocalyx::FuncFuncToCalyxPattern>(typeConverter,
                                                     &getContext());
  // Add return conversion pattern
  patterns.add<lowertocalyx::FuncReturnToCalyxPattern>(typeConverter,
                                                       &getContext());

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    return failure();
  }

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