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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "llvm/Support/MathExtras.h"

namespace circt {
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

mlir::Value resolveDoneSignalForValue(mlir::Value val, mlir::Location loc,
                                      mlir::OpBuilder &builder) {
  using namespace circt::calyx;

  if (auto producerOp = val.getDefiningOp()) {
    // Check if the producer is a calyx.register
    if (auto regOp = dyn_cast<RegisterOp>(producerOp)) {
      return regOp.getDone();
    }

    // Check if the producer is a calyx.seq_mem
    if (auto memOp = dyn_cast<SeqMemoryOp>(producerOp)) {
      return memOp.done();
    }

    // Check for operations with multiple results that might have a done signal
    if (producerOp->getNumResults() > 1) {
      // Look for i1 results that could be done signals
      for (mlir::Value result : producerOp->getResults()) {
        if (result.getType().isSignlessInteger(1) && result != val) {
          // This could be a done signal - return it
          return result;
        }
      }
    }

    // Check if the operation has a calyx.done attribute
    if (producerOp->hasAttr("calyx.done")) {
      // Look for the result marked as done
      for (mlir::Value result : producerOp->getResults()) {
        if (result.getType().isSignlessInteger(1)) {
          return result;
        }
      }
    }

    // For arithmetic operations and other Calyx library ops, check if they
    // follow the standard pattern where the last result is the done signal
    if (producerOp->getNumResults() == 2 &&
        producerOp->getResult(1).getType().isSignlessInteger(1)) {
      return producerOp->getResult(1);
    }

    // For pipelined arithmetic operations (like MultPipeLibOp), check if they
    // have a done signal These have 7 results: clk, reset, go, left, right, out, done
    if (producerOp->getNumResults() == 7 &&
        producerOp->getResult(6).getType().isSignlessInteger(1)) {
      return producerOp->getResult(6);
    }
    
    // Legacy support for 4-result operations (may be deprecated)
    if (producerOp->getNumResults() == 4 &&
        producerOp->getResult(3).getType().isSignlessInteger(1)) {
      return producerOp->getResult(3);
    }
  }

  // Fallback: create constant 1 for immediate values, block arguments, etc.
  auto constantOp = builder.create<hw::ConstantOp>(
      loc, builder.getIntegerAttr(builder.getI1Type(), 1));
  return constantOp.getResult();
}

// Overload that uses constant deduplication
mlir::Value resolveDoneSignalForValue(mlir::Value val, mlir::Location loc,
                                     mlir::OpBuilder &builder,
                                     mlir::Operation *componentOp) {
  using namespace circt::calyx;

  if (auto producerOp = val.getDefiningOp()) {
    // Check if the producer is a calyx.register
    if (auto regOp = dyn_cast<RegisterOp>(producerOp)) {
      return regOp.getDone();
    }

    // Check if the producer is a calyx.seq_mem
    if (auto memOp = dyn_cast<SeqMemoryOp>(producerOp)) {
      return memOp.done();
    }

    // Check for operations with multiple results that might have a done signal
    if (producerOp->getNumResults() > 1) {
      // Look for i1 results that could be done signals
      for (mlir::Value result : producerOp->getResults()) {
        if (result.getType().isSignlessInteger(1) && result != val) {
          // This could be a done signal - return it
          return result;
        }
      }
    }

    // Check if the operation has a calyx.done attribute
    if (producerOp->hasAttr("calyx.done")) {
      // Look for the result marked as done
      for (mlir::Value result : producerOp->getResults()) {
        if (result.getType().isSignlessInteger(1)) {
          return result;
        }
      }
    }

    // For arithmetic operations and other Calyx library ops, check if they
    // follow the standard pattern where the last result is the done signal
    if (producerOp->getNumResults() == 2 &&
        producerOp->getResult(1).getType().isSignlessInteger(1)) {
      return producerOp->getResult(1);
    }

    // For pipelined arithmetic operations (like MultPipeLibOp), check if they
    // have a done signal These have 7 results: clk, reset, go, left, right, out, done
    if (producerOp->getNumResults() == 7 &&
        producerOp->getResult(6).getType().isSignlessInteger(1)) {
      return producerOp->getResult(6);
    }
    
    // Legacy support for 4-result operations (may be deprecated)
    if (producerOp->getNumResults() == 4 &&
        producerOp->getResult(3).getType().isSignlessInteger(1)) {
      return producerOp->getResult(3);
    }
  }

  // Fallback: use constant deduplication for component operations
  return getOrCreateConstant(componentOp, builder, 1);
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

mlir::Value getOrCreateConstant(mlir::Operation *parentOp,
                               mlir::ConversionPatternRewriter &rewriter,
                               int64_t value, 
                               std::optional<unsigned> bitWidth) {
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
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(bodyBlock);

  // Create the constant with specified bit width using parent's location
  auto constantOp = rewriter.create<hw::ConstantOp>(
      loc, rewriter.getIntegerAttr(rewriter.getIntegerType(actualBitWidth), value));
  
  return constantOp.getResult();
}

// Overload for regular OpBuilder
mlir::Value getOrCreateConstant(mlir::Operation *parentOp,
                               mlir::OpBuilder &builder,
                               int64_t value, 
                               std::optional<unsigned> bitWidth) {
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
      loc, builder.getIntegerAttr(builder.getIntegerType(actualBitWidth), value));
  
  return constantOp.getResult();
}

// Example usage of the generic getOrCreateOperation template for SeqMemoryOp deduplication
// This demonstrates how to use the template for any operation type:
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
//       return rewriter.create<calyx::SeqMemoryOp>(loc, expectedMemName, elementWidth, sizes, addrSizes);
//     }
// );

} // namespace lowertocalyx
} // namespace circt