//===- LowerToCalyxUtil.cpp - Utility functions for LowerToCalyx --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for the LowerToCalyx conversion.
//
//===----------------------------------------------------------------------===//

#include "LowerToCalyxUtil.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "llvm/Support/MathExtras.h"

namespace circt {

using namespace calyx;
namespace lowertocalyx {

void moveRegionContents(mlir::Region &sourceRegion, mlir::Region &targetRegion,
                        mlir::ConversionPatternRewriter &rewriter) {
  if (sourceRegion.empty()) {
    return;
  }

  // Ensure target region has exactly one block
  if (!targetRegion.hasOneBlock()) {
    return;
  }

  // Ensure source region has exactly one block
  if (!sourceRegion.hasOneBlock()) {
    return;
  }

  mlir::Block *sourceBlock = &sourceRegion.front();
  mlir::Block *targetBlock = &targetRegion.front();

  // Get the arguments from target block to replace source block arguments
  llvm::SmallVector<mlir::Value> argValues;
  for (unsigned i = 0;
       i < sourceBlock->getNumArguments() && i < targetBlock->getNumArguments();
       ++i) {
    argValues.push_back(targetBlock->getArgument(i));
  }

  // Use inlineBlockBefore to move operations from source block to target block
  // This should avoid creating extra blocks
  rewriter.inlineBlockBefore(sourceBlock, targetBlock, targetBlock->end(),
                             argValues);
}

mlir::DictionaryAttr getMandatoryPortAttr(mlir::MLIRContext *ctx,
                                          llvm::StringRef name) {
  using namespace mlir;
  return DictionaryAttr::get(
      ctx, {NamedAttribute(StringAttr::get(ctx, name), UnitAttr::get(ctx))});
}

void addMandatoryComponentPorts(
    mlir::PatternRewriter &rewriter,
    llvm::SmallVectorImpl<circt::calyx::PortInfo> &ports) {
  using namespace circt::calyx;
  mlir::MLIRContext *ctx = rewriter.getContext();
  ports.push_back({
      rewriter.getStringAttr(clkPort),
      rewriter.getI1Type(),
      Direction::Input,
      getMandatoryPortAttr(ctx, clkPort),
  });
  ports.push_back({
      rewriter.getStringAttr(resetPort),
      rewriter.getI1Type(),
      Direction::Input,
      getMandatoryPortAttr(ctx, resetPort),
  });
  ports.push_back({
      rewriter.getStringAttr(goPort),
      rewriter.getI1Type(),
      Direction::Input,
      getMandatoryPortAttr(ctx, goPort),
  });
  ports.push_back({
      rewriter.getStringAttr(donePort),
      rewriter.getI1Type(),
      Direction::Output,
      getMandatoryPortAttr(ctx, donePort),
  });
}

std::string getOpUniqueName(mlir::Operation *op) {
  // Get the operation name (without dialect prefix)
  std::string opName = op->getName().getStringRef().str();
  size_t dotPos = opName.find('.');
  if (dotPos != std::string::npos) {
    opName = opName.substr(dotPos + 1);
  }

  // // Try to get a simple name from the first SSA result
  // if (op->getNumResults() > 0) {
  //   auto result = op->getResult(0);
  //   std::string valueName;
  //   llvm::raw_string_ostream os(valueName);
  //   mlir::OpPrintingFlags flags;
  //   result.printAsOperand(os, flags);
  //   os.flush();

  //   // If we got a simple SSA value like %6, use opName_6
  //   if (valueName.size() > 1 && valueName[0] == '%' &&
  //       std::all_of(valueName.begin() + 1, valueName.end(), ::isdigit)) {
  //     return opName + "_" + valueName.substr(1);
  //   }
  // }

  // Fallback 1: Try to use line number from location
  if (auto fileLoc = dyn_cast<mlir::FileLineColLoc>(op->getLoc())) {
    return opName + "_line_" + std::to_string(fileLoc.getLine());
  } else if (auto fusedLoc = dyn_cast<mlir::FusedLoc>(op->getLoc())) {
    // Try to extract line number from fused location
    for (auto loc : fusedLoc.getLocations()) {
      if (auto fileLoc = dyn_cast<mlir::FileLineColLoc>(loc)) {
        return opName + "_line_" + std::to_string(fileLoc.getLine());
      }
    }
  }

  // Fallback 2: Use memory address as last resort
  return opName + "_" + std::to_string(reinterpret_cast<uintptr_t>(op));
}

// Helper function to check if a value originates from a constant or parent
// argument
bool isTerminalValue(mlir::Value val, mlir::Operation *parentOp) {
  // Check if value is a block argument (comes from parent operation)
  if (auto blockArg = dyn_cast<mlir::BlockArgument>(val)) {
    return true;
  }

  // Check if value is produced by a constant operation
  if (auto producerOp = val.getDefiningOp()) {
    if (isa<hw::ConstantOp>(producerOp) ||
        isa<mlir::arith::ConstantOp>(producerOp)) {
      return true;
    }
  }

  return false;
}

// Helper function to recursively track dependency chain and return done signal
mlir::Value trackAssignUsersRecursively(mlir::Value val, mlir::Operation *parentOp,
                                        llvm::SmallPtrSet<mlir::Value, 16> &visited) {

  // Avoid infinite recursion
  if (visited.contains(val)) {
    return getOrCreateConstant(parentOp, 1);
  }
  visited.insert(val);

  // Check termination conditions first
  if (isTerminalValue(val, parentOp)) {
    return getOrCreateConstant(parentOp, 1); // Terminal value found, return constant true
  }

  if (auto producerOp = val.getDefiningOp()) {
    // Check if this operation has pipeline or sequential traits
    if (producerOp->hasTrait<calyx::SequentialTrait>() ||
        producerOp->hasTrait<calyx::PipelineTrait>()) {
      if (auto cellOp = dyn_cast<circt::calyx::CellInterface>(producerOp)) {
        auto portAttrs = cellOp.portAttributes();
        auto results = producerOp->getResults();

        // Find the result that corresponds to the done port
        for (size_t i = 0; i < results.size() && i < portAttrs.size(); ++i) {
          if (portAttrs[i].contains(donePort)) {
            return results[i]; // Found pipelined/sequential done signal
          }
        }
      }
    }
  }

  // Track all assign operations that use this value
  for (auto &use : val.getUses()) {
    auto *userOp = use.getOwner();

    // Check if the user is an assign operation
    if (auto assignOp = dyn_cast<circt::calyx::AssignOp>(userOp)) {
      // Get the source value of the assign operation
      mlir::Value srcVal = assignOp.getSrc();

      // Recursively check the source value's producer
      // If source is terminal, we have a path to constant/argument
      if (isTerminalValue(srcVal, parentOp)) {
        return getOrCreateConstant(parentOp, 1);
      } else {
        // Recursively track the source producer
        return trackAssignUsersRecursively(srcVal, parentOp, visited);
      }
    }
  }

  return getOrCreateConstant(parentOp, 1);
}

// Helper function to extract done port from pipeline/sequential operations
mlir::Value getOperationDonePort(mlir::Operation *producerOp) {
  if (auto cellOp = dyn_cast<circt::calyx::CellInterface>(producerOp)) {
    auto portAttrs = cellOp.portAttributes();
    auto results = producerOp->getResults();

    // Find the result that corresponds to the done port
    for (size_t i = 0; i < results.size() && i < portAttrs.size(); ++i) {
      if (portAttrs[i].contains(donePort)) {
        return results[i];
      }
    }
  }
  return nullptr;
}

// Helper function to create AND gate for binary operations
mlir::Value createAndGate(mlir::Value leftDone, mlir::Value rightDone,
                          mlir::Operation *producerOp, mlir::Operation *parentOp) {
  mlir::Block *bodyBlock = nullptr;
  mlir::Location loc = parentOp->getLoc();

  if (auto funcOp = dyn_cast<mlir::func::FuncOp>(parentOp)) {
    bodyBlock = &funcOp.getBody().front();
  } else if (auto componentOp = dyn_cast<circt::calyx::ComponentOp>(parentOp)) {
    bodyBlock = componentOp.getBodyBlock();
  } else if (!parentOp->getRegions().empty() && !parentOp->getRegion(0).empty()) {
    bodyBlock = &parentOp->getRegion(0).front();
  } else {
    return getOrCreateConstant(parentOp, 1);
  }

  OpBuilder compBuilder(parentOp->getContext());
  compBuilder.setInsertionPoint(parentOp);

  calyx::WiresOp wiresOp = *bodyBlock->getOps<calyx::WiresOp>().begin();
  auto &wiresBlock = wiresOp.getBodyRegion().front();
  OpBuilder wiresBuilder(&wiresBlock, wiresBlock.end());

  auto andOp = compBuilder.create<calyx::AndLibOp>(
      loc, compBuilder.getStringAttr(getOpUniqueName(producerOp)),
      compBuilder.getI1Type());
  wiresBuilder.create<calyx::AssignOp>(loc, andOp.getLeft(), leftDone);
  wiresBuilder.create<calyx::AssignOp>(loc, andOp.getRight(), rightDone);
  return andOp.getOut();
}

// Public interface that classifies operations and handles done signals appropriately
mlir::Value resolveDoneSignalForValue(mlir::Value val, mlir::Operation *parentOp) {
  llvm::SmallPtrSet<mlir::Value, 16> visitedValues;

  if (auto producerOp = val.getDefiningOp()) {
    // Case 1: Direct pipeline/sequential operation - return its done port
    if (producerOp->hasTrait<calyx::SequentialTrait>() ||
        producerOp->hasTrait<calyx::PipelineTrait>()) {
      if (auto donePort = getOperationDonePort(producerOp)) {
        return donePort;
      }
    }

    // Case 2: Binary combinational operation - AND both operand done signals
    if (producerOp->hasTrait<calyx::Combinational>() &&
        producerOp->hasTrait<calyx::BinaryOpTrait>()) {
      if (producerOp->getNumOperands() >= 2) {
        llvm::SmallPtrSet<mlir::Value, 16> leftVisited;
        llvm::SmallPtrSet<mlir::Value, 16> rightVisited;
        
        mlir::Value leftDone = trackAssignUsersRecursively(producerOp->getOperand(0), parentOp, leftVisited);
        mlir::Value rightDone = trackAssignUsersRecursively(producerOp->getOperand(1), parentOp, rightVisited);
        
        return createAndGate(leftDone, rightDone, producerOp, parentOp);
      } else {
        return getOrCreateConstant(parentOp, 1); // No operands
      }
    }

    // Case 3: Unary combinational operation - track input
    if (producerOp->hasTrait<calyx::UnaryOpTrait>()) {
      if (producerOp->getNumOperands() >= 1) {
        return trackAssignUsersRecursively(producerOp->getOperand(0), parentOp, visitedValues);
      } else {
        return getOrCreateConstant(parentOp, 1); // No operands
      }
    }
  }

  // Case 4: Other operations - use dependency tracking
  return trackAssignUsersRecursively(val, parentOp, visitedValues);
}

void updateComponentDoneConnection(circt::calyx::ComponentOp comp,
                                   mlir::Value newDone, mlir::Location loc,
                                   mlir::OpBuilder &builder) {
  using namespace circt::calyx;

  // Find the component's done port
  mlir::Value donePort = comp.getDonePort();
  if (!donePort) {
    return; // Component doesn't have a done port
  }

  // Find the wires operation to create the assignment
  auto wiresOp = *comp.getBodyBlock()->getOps<WiresOp>().begin();
  auto &wiresBlock = wiresOp.getBodyRegion().front();

  // Remove any existing assignments to the done port
  llvm::SmallVector<AssignOp> toErase;
  for (auto assignOp : wiresBlock.getOps<AssignOp>()) {
    if (assignOp.getDest() == donePort) {
      toErase.push_back(assignOp);
    }
  }
  for (auto assignOp : toErase) {
    assignOp.erase();
  }

  // Create new assignment to connect the done port to the new done signal
  mlir::OpBuilder wiresBuilder(&wiresBlock, wiresBlock.end());
  wiresBuilder.create<AssignOp>(loc, donePort, newDone);
}

// Example usage of the generic getOrCreateOperation template for SeqMemoryOp
// deduplication This demonstrates how to use the template for any operation
// type:
//
// auto memOp = getOrCreateOperation<calyx::SeqMemoryOp>(
//     componentOp.getOperation(),
//     rewriter,
//     loc,
//     [&](calyx::SeqMemoryOp existingMem) -> bool {
//       // Match criteria: same name and dimensions
//       return existingMem.getSymName() == expectedMemName &&
//              existingMem.getSizes() == expectedSizes;
//     },
//     [&]() -> calyx::SeqMemoryOp {
//       // Creator function: create new memory if none matches
//       return rewriter.create<calyx::SeqMemoryOp>(loc, expectedMemName,
//       elementWidth, sizes, addrSizes);
//     }
// );

/// Unified utility functions implementation

mlir::Value getOrCreateConstant(mlir::Operation *parentOp, int64_t value,
                                std::optional<unsigned> bitWidth) {
  // Always set insertion point at the parentOp
  mlir::OpBuilder builder(parentOp->getContext());
  builder.setInsertionPoint(parentOp);

  // Calculate the actual bit width needed
  unsigned actualBitWidth = bitWidth.value_or(1); // Default to 1 bit for now

  // Get the body block from either func::FuncOp or calyx::ComponentOp
  mlir::Block *bodyBlock = nullptr;
  mlir::Location loc = parentOp->getLoc();

  if (auto funcOp = dyn_cast<mlir::func::FuncOp>(parentOp)) {
    bodyBlock = &funcOp.getBody().front();
  } else if (auto componentOp = dyn_cast<circt::calyx::ComponentOp>(parentOp)) {
    bodyBlock = componentOp.getBodyBlock();
  } else {
    // Fallback: try to get the first region's first block
    if (!parentOp->getRegions().empty() && !parentOp->getRegion(0).empty()) {
      bodyBlock = &parentOp->getRegion(0).front();
    } else {
      return nullptr; // Cannot determine body block
    }
  }

  if (!bodyBlock) {
    return nullptr;
  }

  // Search for existing constant with the same value and bit width in the body
  for (auto &op : bodyBlock->getOperations()) {
    if (auto constOp = dyn_cast<hw::ConstantOp>(op)) {
      // Compare APInt values and bit widths
      const auto &constValue = constOp.getValue();
      if (constValue.getBitWidth() == actualBitWidth &&
          constValue.getZExtValue() == static_cast<uint64_t>(value)) {
        return constOp.getResult();
      }
    }
  }

  // If constant doesn't exist, create it at the parent operation level
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(bodyBlock);

  // Create the constant with specified bit width using parent's location
  auto constantOp = builder.create<hw::ConstantOp>(
      loc,
      builder.getIntegerAttr(builder.getIntegerType(actualBitWidth), value));

  return constantOp.getResult();
}

mlir::Value createReg(mlir::Operation *parentOp, mlir::Type type,
                      llvm::StringRef name) {
  mlir::OpBuilder builder(parentOp->getContext());
  builder.setInsertionPoint(parentOp);
  mlir::Location loc = parentOp->getLoc();

  // Generate unique name if not provided
  std::string regName =
      name.empty() ? ("reg_" + getOpUniqueName(parentOp)) : name.str();

  // Get the bit width from the type
  unsigned bitWidth = 1;
  if (auto intType = dyn_cast<mlir::IntegerType>(type)) {
    bitWidth = intType.getWidth();
  }

  // Handle different parent operation types
  if (auto componentOp = dyn_cast<circt::calyx::ComponentOp>(parentOp)) {
    // Set insertion point to the component body block start
    builder.setInsertionPointToStart(componentOp.getBodyBlock());

    // Create the register directly in the component body
    auto regOp = builder.create<circt::calyx::RegisterOp>(
        loc, builder.getStringAttr(regName), bitWidth);
    return regOp.getOut();
  }

  // Fallback for other operation types (function context)
  auto regOp = builder.create<circt::calyx::RegisterOp>(
      loc, builder.getStringAttr(regName), bitWidth);
  return regOp.getOut();
}

} // namespace lowertocalyx
} // namespace circt