//===- MemrefToCalyx.cpp - Memory to Calyx Conversion ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementations for converting memref operations
// to Calyx.
//
//===----------------------------------------------------------------------===//

#include "LowerToCalyxUtil.h"
#include "circt/Dialect/Calyx/CalyxLoweringUtils.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "convertPattern.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// Forward declaration from LowerToCalyx.cpp
namespace circt {
namespace lowertocalyx {
std::string getOpUniqueName(mlir::Operation *op);
}
} // namespace circt

using namespace circt;
using namespace circt::lowertocalyx;
using namespace mlir;

namespace circt {
namespace lowertocalyx {

/// Complete memref.load to Calyx conversion pattern
/// This pattern implements the full load operation conversion based on SCF to
/// Calyx approach Helper function to find or create a memory operation for
/// function argument memrefs
static calyx::SeqMemoryOp
getOrCreateFunctionArgMemory(Value memref, calyx::ComponentOp componentOp,
                             ConversionPatternRewriter &rewriter,
                             Location loc) {
  auto memrefType = cast<MemRefType>(memref.getType());
  if (!memrefType.hasStaticShape()) {
    return nullptr;
  }

  // Only accept 1-dimensional memrefs
  auto shape = memrefType.getShape();
  if (shape.size() != 1) {
    return nullptr;
  }

  // Look for existing memory operation for this function argument
  auto blockArg = cast<BlockArgument>(memref);
  std::string expectedMemName =
      "mem_arg_" + std::to_string(blockArg.getArgNumber());

  // Search for the memory operation in the component
  for (auto &op : componentOp.getBodyBlock()->getOperations()) {
    if (auto seqMemOp = dyn_cast<calyx::SeqMemoryOp>(op)) {
      if (seqMemOp.getSymName() == expectedMemName) {
        return seqMemOp;
      }
    }
  }

  // If memory doesn't exist, create it on-demand at component level
  auto elementType = memrefType.getElementType();

  if (!elementType.isSignlessInteger()) {
    return nullptr;
  }

  auto intType = cast<IntegerType>(elementType);
  int64_t elementWidth = intType.getWidth();

  // Calculate sizes and address widths for the single dimension
  SmallVector<int64_t> sizes, addrSizes;
  int64_t dim = shape[0];
  sizes.push_back(dim);
  addrSizes.push_back(llvm::Log2_64_Ceil(dim));

  // Create the memory at the component level (before wires)
  OpBuilder::InsertionGuard memGuard(rewriter);
  rewriter.setInsertionPointToStart(componentOp.getBodyBlock());

  auto memOp = rewriter.create<calyx::SeqMemoryOp>(
      loc, expectedMemName, elementWidth, sizes, addrSizes);

  // Set external attribute for proper Calyx compilation
  memOp->setAttr("external", rewriter.getIntegerAttr(rewriter.getI1Type(), 1));

  return memOp;
}

LogicalResult MemrefLoadToCalyxPattern::matchAndRewrite(
    memref::LoadOp loadOp, memref::LoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = loadOp.getLoc();
  Value memref = adaptor.getMemref();
  auto indices = adaptor.getIndices();

  // Find the parent component
  auto componentOp = loadOp->getParentOfType<calyx::ComponentOp>();
  if (!componentOp) {
    return rewriter.notifyMatchFailure(
        loadOp, "Load operation not within a calyx.component");
  }

  // Find the parent group where this load operation currently resides
  auto parentGroup = loadOp->getParentOfType<calyx::GroupOp>();
  if (!parentGroup) {
    return rewriter.notifyMatchFailure(
        loadOp, "Load operation not within a calyx.group");
  }

  // Find the memory operation for this memref
  calyx::SeqMemoryOp memOp = nullptr;
  auto definingOp = memref.getDefiningOp();

  if (definingOp) {
    // Case 1a: memref defined by a seq_mem operation directly
    memOp = dyn_cast<calyx::SeqMemoryOp>(definingOp);
  } else {
    // Case 2: memref is a block argument (function parameter)
    // Check if memref is 1-dimensional before attempting conversion
    auto memrefType = cast<MemRefType>(memref.getType());
    if (memrefType.getShape().size() != 1) {
      return rewriter.notifyMatchFailure(
          loadOp, "Multi-dimensional memrefs not supported. Please run memref "
                  "flattening pass before lowering to Calyx");
    }

    memOp = getOrCreateFunctionArgMemory(memref, componentOp, rewriter, loc);
    if (!memOp) {
      return rewriter.notifyMatchFailure(
          loadOp, "Failed to create memory for function argument");
    }
  }

  // Create assignment for the single address port (1D memref has only one
  // index)
  Value addrPort = memOp.addrPort(0);
  Value indexValue = indices[0];
  Value addrOut = nullptr;

  // Create continuous assignment in wires section for address
  auto wiresOp = componentOp.getWiresOp();
  OpBuilder wiresBuilder(wiresOp.getBodyBlock(), wiresOp.getBodyBlock()->end());

  // Handle type conversion - cast i32 index to i4 address
  if (addrPort.getType() != indexValue.getType()) {
    auto addrPortType = cast<IntegerType>(addrPort.getType());

    // Create Calyx slice operation directly instead of arith.trunci
    std::string sliceName =
        "slice_addr_" + getOpUniqueName(loadOp.getOperation());

    // Create slice operation at component level
    OpBuilder::InsertionGuard sliceGuard(rewriter);
    rewriter.setInsertionPointToStart(componentOp.getBodyBlock());

    // SliceLibOp takes input type and output type
    SmallVector<Type> sliceTypes = {indexValue.getType(), addrPortType};
    auto sliceOp = rewriter.create<calyx::SliceLibOp>(
        loc, rewriter.getStringAttr(sliceName), sliceTypes);

    // Assign input to slice and slice output to address port
    wiresBuilder.create<calyx::AssignOp>(loc, sliceOp.getIn(), indexValue);
    addrOut = sliceOp.getOut();
  } else {
    addrOut = indexValue;
  }

  // Now work within the existing group where the load operation was located
  rewriter.setInsertionPoint(loadOp);

  // Create continuous assignment for address port
  rewriter.create<calyx::AssignOp>(loc, addrPort, addrOut);

  // Determine content enable signal based on address source
  Value contentEnableSignal =
      resolveDoneSignalForValue(indexValue, loc, rewriter, componentOp);

  // Connect content enable signal to memory within the group
  rewriter.create<calyx::AssignOp>(loc, memOp.contentEn(), contentEnableSignal);

  // Update the group's done signal to use memory done signal
  // Find existing group_done and replace it with memory done
  for (auto &op : parentGroup.getBodyBlock()->getOperations()) {
    if (auto groupDoneOp = dyn_cast<calyx::GroupDoneOp>(op)) {
      OpBuilder::InsertionGuard doneGuard(rewriter);
      rewriter.setInsertionPoint(groupDoneOp);
      rewriter.replaceOpWithNewOp<calyx::GroupDoneOp>(groupDoneOp,
                                                      memOp.done());
      break;
    }
  }

  // Replace the load operation result with memory read data directly
  // SCF to Calyx will handle register creation if needed for multiple loads
  rewriter.replaceOp(loadOp, memOp.readData());

  return success();
}

/// Complete memref.store to Calyx conversion pattern
/// This pattern implements the full store operation conversion based on SCF to
/// Calyx approach
LogicalResult MemrefStoreToCalyxPattern::matchAndRewrite(
    memref::StoreOp storeOp, memref::StoreOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = storeOp.getLoc();
  Value valueToStore = adaptor.getValue();
  Value memref = adaptor.getMemref();
  auto indices = adaptor.getIndices();

  // Find the parent component
  auto componentOp = storeOp->getParentOfType<calyx::ComponentOp>();
  if (!componentOp) {
    return rewriter.notifyMatchFailure(
        storeOp, "Store operation not within a calyx.component");
  }

  // Find the parent group where this store operation currently resides
  auto parentGroup = storeOp->getParentOfType<calyx::GroupOp>();
  if (!parentGroup) {
    return rewriter.notifyMatchFailure(
        storeOp, "Store operation not within a calyx.group");
  }

  // Find the memory operation for this memref
  calyx::SeqMemoryOp memOp = nullptr;
  auto definingOp = memref.getDefiningOp();

  if (definingOp) {
    // Case 1a: memref defined by a seq_mem operation directly
    memOp = dyn_cast<calyx::SeqMemoryOp>(definingOp);
  } else {
    // Case 2: memref is a block argument (function parameter)
    // Check if memref is 1-dimensional before attempting conversion
    auto memrefType = cast<MemRefType>(memref.getType());
    if (memrefType.getShape().size() != 1) {
      return rewriter.notifyMatchFailure(
          storeOp, "Multi-dimensional memrefs not supported. Please run memref "
                   "flattening pass before lowering to Calyx");
    }

    memOp = getOrCreateFunctionArgMemory(memref, componentOp, rewriter, loc);
    if (!memOp) {
      return rewriter.notifyMatchFailure(
          storeOp, "Failed to create memory for function argument");
    }
  }

  // Now work within the existing group where the store operation was located
  rewriter.setInsertionPoint(storeOp);

  // Create assignment for the single address port (1D memref has only one
  // index)
  Value addrPort = memOp.addrPort(0);
  Value indexValue = indices[0];
  Value addrOut = nullptr;

  // Create continuous assignment in wires section for address
  auto wiresOp = componentOp.getWiresOp();
  OpBuilder wiresBuilder(wiresOp.getBodyBlock(), wiresOp.getBodyBlock()->end());

  // Handle type conversion - cast i32 index to i4 address
  if (addrPort.getType() != indexValue.getType()) {
    auto addrPortType = cast<IntegerType>(addrPort.getType());

    // Create Calyx slice operation directly instead of arith.trunci
    std::string sliceName =
        "slice_addr_" + getOpUniqueName(storeOp.getOperation());

    // Create slice operation at component level
    OpBuilder::InsertionGuard sliceGuard(rewriter);
    rewriter.setInsertionPointToStart(componentOp.getBodyBlock());

    // SliceLibOp takes input type and output type
    SmallVector<Type> sliceTypes = {indexValue.getType(), addrPortType};
    auto sliceOp = rewriter.create<calyx::SliceLibOp>(
        loc, rewriter.getStringAttr(sliceName), sliceTypes);

    // Assign input to slice and slice output to address port
    wiresBuilder.create<calyx::AssignOp>(loc, sliceOp.getIn(), indexValue);
    addrOut = sliceOp.getOut();
  } else {
    addrOut = indexValue;
  }

  // Connect value to memory write data within the existing group
  rewriter.create<calyx::AssignOp>(loc, addrPort, addrOut);
  rewriter.create<calyx::AssignOp>(loc, memOp.writeData(), valueToStore);

  // Determine write enable signal based on source data dependencies
  Value writeEnableSignal =
      resolveDoneSignalForValue(valueToStore, loc, rewriter, componentOp);

  // Connect write enable signal within the existing group
  rewriter.create<calyx::AssignOp>(loc, memOp.writeEn(), writeEnableSignal);

  // Update the group's done signal to use memory done signal
  // Find existing group_done and replace it with memory done
  for (auto &op : parentGroup.getBodyBlock()->getOperations()) {
    if (auto groupDoneOp = dyn_cast<calyx::GroupDoneOp>(op)) {
      OpBuilder::InsertionGuard doneGuard(rewriter);
      rewriter.setInsertionPoint(groupDoneOp);
      rewriter.replaceOpWithNewOp<calyx::GroupDoneOp>(groupDoneOp,
                                                      memOp.done());
      break;
    }
  }

  // Store operations don't have results, so just erase the original
  rewriter.eraseOp(storeOp);

  return success();
}

/// Complete memref.alloca to Calyx conversion pattern
/// This pattern converts local memory allocations to Calyx sequential memory
LogicalResult MemrefAllocaToCalyxPattern::matchAndRewrite(
    memref::AllocaOp allocaOp, memref::AllocaOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = allocaOp.getLoc();
  auto memrefType = cast<MemRefType>(allocaOp.getType());

  // Get memory size and element width
  if (!memrefType.hasStaticShape()) {
    return rewriter.notifyMatchFailure(allocaOp,
                                       "Dynamic memory shapes not supported");
  }

  auto shape = memrefType.getShape();

  // Only accept 1-dimensional memrefs
  if (shape.size() != 1) {
    return rewriter.notifyMatchFailure(
        allocaOp, "Multi-dimensional memrefs not supported. Please run memref "
                  "flattening pass before lowering to Calyx");
  }

  auto elementType = memrefType.getElementType();

  if (!elementType.isSignlessInteger()) {
    return rewriter.notifyMatchFailure(
        allocaOp, "Only signless integer element types supported");
  }

  auto intType = cast<IntegerType>(elementType);
  int64_t elementWidth = intType.getWidth();

  // Find the parent component to create memory at the component level
  auto componentOp = allocaOp->getParentOfType<calyx::ComponentOp>();
  if (!componentOp) {
    return rewriter.notifyMatchFailure(
        allocaOp, "Memory allocation not within a calyx.component");
  }

  // Create the memory at the component level (before wires section)
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(componentOp.getBodyBlock());

  // Create a unique memory name
  std::string memName =
      "mem_" +
      std::to_string(reinterpret_cast<uintptr_t>(allocaOp.getOperation()));

  // Calculate sizes and address widths for each dimension
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> addrSizes;

  for (auto dim : shape) {
    sizes.push_back(dim);
    addrSizes.push_back(llvm::Log2_64_Ceil(dim));
  }

  // Create the Calyx sequential memory operation
  auto memOp = rewriter.create<calyx::SeqMemoryOp>(loc, memName, elementWidth,
                                                   sizes, addrSizes);

  // NOTE: Do NOT set external attribute for alloca memory operations
  // Unlike function arguments, alloca creates internal/local memory
  // The external attribute is only for function parameters that represent
  // external memory interfaces

  // Create a constant handle to the memory operation for load/store patterns to
  // find This approach allows load/store patterns to map the constant back to
  // the memory operation
  auto memoryId = reinterpret_cast<uintptr_t>(memOp.getOperation());
  auto memHandleConst = rewriter.create<hw::ConstantOp>(
      loc, rewriter.getIntegerAttr(rewriter.getIntegerType(64), memoryId));

  // Replace the alloca with this constant "handle"
  rewriter.replaceOp(allocaOp, memHandleConst.getResult());

  return success();
}

/// Complete function argument memref preparation pattern
/// This pattern creates memory operations for function argument memrefs in
/// components
LogicalResult MemrefFunctionArgPattern::matchAndRewrite(
    calyx::ComponentOp componentOp, calyx::ComponentOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Create memory operations for function argument memrefs
  for (auto arg : componentOp.getArguments()) {
    if (auto memrefType = dyn_cast<MemRefType>(arg.getType())) {
      if (!memrefType.hasStaticShape()) {
        continue; // Skip dynamic shapes
      }

      // Calculate memory parameters
      auto shape = memrefType.getShape();
      int64_t elementWidth = memrefType.getElementTypeBitWidth();

      SmallVector<int64_t> sizes, addrSizes;
      for (auto size : shape) {
        sizes.push_back(size);
        addrSizes.push_back(llvm::Log2_64_Ceil(size));
      }

      // Create unique memory name for function argument
      auto blockArg = cast<BlockArgument>(arg);
      std::string memName =
          "mem_arg_" + std::to_string(blockArg.getArgNumber());

      // Check if memory already exists
      bool memoryExists = false;
      for (auto &op : componentOp.getBodyBlock()->getOperations()) {
        if (auto seqMemOp = dyn_cast<calyx::SeqMemoryOp>(op)) {
          if (seqMemOp.getSymName() == memName) {
            memoryExists = true;
            break;
          }
        }
      }

      if (!memoryExists) {
        // Insert memory operation at the beginning of component body
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(componentOp.getBodyBlock());

        // Create SeqMemoryOp
        auto memOp = rewriter.create<calyx::SeqMemoryOp>(
            componentOp.getLoc(), memName, elementWidth, sizes, addrSizes);

        // Set external attribute for memory operations
        memOp->setAttr("external",
                       rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      }
    }
  }

  return success();
}

// Implementation for memref.load pattern
LogicalResult MemoryLoadToCalyxPattern::matchAndRewrite(
    mlir::memref::LoadOp op, mlir::memref::LoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto loc = op.getLoc();
  Value memref = adaptor.getMemref();
  auto indices = adaptor.getIndices();

  // Find the parent component
  auto componentOp = op->getParentOfType<calyx::ComponentOp>();
  if (!componentOp) {
    return rewriter.notifyMatchFailure(
        op, "Load operation not within a calyx.component");
  }

  // The memref should be either a calyx.seq_mem operation result or a component
  // argument
  auto definingOp = memref.getDefiningOp();
  calyx::SeqMemoryOp memOp = nullptr;

  if (definingOp) {
    // Case 1: memref defined by an operation (memref.alloc/alloca ->
    // calyx.seq_mem)
    memOp = dyn_cast<calyx::SeqMemoryOp>(definingOp);
    if (!memOp) {
      return rewriter.notifyMatchFailure(
          op, "Memref defined by unsupported operation");
    }
  } else {
    // Case 2: memref is a block argument (function parameter)
    // Create a placeholder memory operation for the argument
    auto memrefType = cast<MemRefType>(memref.getType());

    if (!memrefType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "Dynamic memory shapes not supported");
    }

    auto shape = memrefType.getShape();
    auto elementType = memrefType.getElementType();

    if (!elementType.isSignlessInteger()) {
      return rewriter.notifyMatchFailure(
          op, "Only signless integer element types supported");
    }

    auto intType = cast<IntegerType>(elementType);
    int64_t elementWidth = intType.getWidth();

    // Create the memory at the component level (before wires section)
    auto wiresOp =
        *componentOp.getBodyBlock()->getOps<calyx::WiresOp>().begin();
    OpBuilder componentBuilder(rewriter.getContext());
    componentBuilder.setInsertionPoint(wiresOp);

    // Create a unique memory name based on the argument
    std::string memName =
        "mem_arg_" + std::to_string(reinterpret_cast<uintptr_t>(
                         memref.getAsOpaquePointer()));

    // Calculate sizes and address widths for each dimension
    SmallVector<int64_t> sizes;
    SmallVector<int64_t> addrSizes;

    for (auto dim : shape) {
      sizes.push_back(dim);
      // Calculate address width: ceil(log2(dim))
      int64_t addrWidth = 1;
      int64_t temp = dim - 1;
      while (temp > 0) {
        temp >>= 1;
        addrWidth++;
      }
      addrSizes.push_back(addrWidth);
    }

    // Create the Calyx sequential memory operation for the argument
    memOp = componentBuilder.create<calyx::SeqMemoryOp>(
        loc, memName, elementWidth, sizes, addrSizes);

    // Set external attribute for proper Calyx compilation (following SCFToCalyx
    // pattern)
    memOp->setAttr("external", componentBuilder.getIntegerAttr(
                                   componentBuilder.getI1Type(), 1));
  }

  // Find the wires operation to create assignments
  auto wiresOp = *componentOp.getBodyBlock()->getOps<calyx::WiresOp>().begin();
  auto &wiresBlock = wiresOp.getBodyRegion().front();
  OpBuilder wiresBuilder(&wiresBlock, wiresBlock.end());

  // Create assignments for address ports
  for (size_t i = 0; i < indices.size(); ++i) {
    Value addrPort = memOp.addrPort(i);
    wiresBuilder.create<calyx::AssignOp>(loc, addrPort, indices[i]);
  }

  // The load result is the memory's read_data port
  Value readData = memOp.readData();
  rewriter.replaceOp(op, readData);

  return success();
}

// Implementation for memref.store pattern
LogicalResult MemoryStoreToCalyxPattern::matchAndRewrite(
    mlir::memref::StoreOp op, mlir::memref::StoreOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto loc = op.getLoc();
  Value valueToStore = adaptor.getValue();
  Value memref = adaptor.getMemref();
  auto indices = adaptor.getIndices();

  // Find the parent component
  auto componentOp = op->getParentOfType<calyx::ComponentOp>();
  if (!componentOp) {
    return rewriter.notifyMatchFailure(
        op, "Store operation not within a calyx.component");
  }

  // The memref should be a calyx.seq_mem operation result
  auto definingOp = memref.getDefiningOp();
  auto memOp = dyn_cast_or_null<calyx::SeqMemoryOp>(definingOp);
  if (!memOp) {
    return rewriter.notifyMatchFailure(op,
                                       "Memref not defined by calyx.seq_mem");
  }

  // Find the wires operation to create assignments
  auto wiresOp = *componentOp.getBodyBlock()->getOps<calyx::WiresOp>().begin();
  auto &wiresBlock = wiresOp.getBodyRegion().front();
  OpBuilder wiresBuilder(&wiresBlock, wiresBlock.end());

  // Create assignments for address ports
  for (size_t i = 0; i < indices.size(); ++i) {
    Value addrPort = memOp.addrPort(i);
    wiresBuilder.create<calyx::AssignOp>(loc, addrPort, indices[i]);
  }

  // Create assignment for write data
  Value writeData = memOp.writeData();
  wiresBuilder.create<calyx::AssignOp>(loc, writeData, valueToStore);

  // Sophisticated write enable control based on the producer of the write value
  Value writeEnableSignal;

  if (auto producerOp = valueToStore.getDefiningOp()) {
    // Check if the producer is a register (calyx.register or calyx.std_reg)
    if (auto regOp = dyn_cast<calyx::RegisterOp>(producerOp)) {
      // Connect write enable to the register's done signal
      writeEnableSignal = regOp.getDone();
    } else {
      // For other operations (arithmetic, constants, etc.), check if they have
      // a done signal
      if (producerOp->hasAttr("calyx.done") ||
          producerOp->getNumResults() > 1) {
        // Try to find a done signal in the results
        for (Value result : producerOp->getResults()) {
          if (result.getType().isSignlessInteger(1)) {
            // Assume this is a done signal
            writeEnableSignal = result;
            break;
          }
        }
      }

      // Fallback: use constant 1 for immediate values (constants, arguments)
      if (!writeEnableSignal) {
        writeEnableSignal =
            getOrCreateConstant(componentOp.getOperation(), rewriter, 1);
      }
    }
  } else {
    // Value is a block argument (function parameter) - use constant 1
    writeEnableSignal =
        getOrCreateConstant(componentOp.getOperation(), rewriter, 1);
  }

  Value writeEn = memOp.writeEn();
  wiresBuilder.create<calyx::AssignOp>(loc, writeEn, writeEnableSignal);

  // Store operations don't produce results, so just erase the original
  rewriter.eraseOp(op);

  return success();
}

// Index type conversion pattern implementation
LogicalResult FuncOpIndexConversionPattern::matchAndRewrite(
    func::FuncOp funcOp, func::FuncOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  
  // Check if function needs index type conversion
  auto funcType = funcOp.getFunctionType();
  bool hasIndexTypes = false;
  
  // Check for index types in inputs and results
  for (Type inputType : funcType.getInputs()) {
    if (inputType.isIndex()) {
      hasIndexTypes = true;
      break;
    }
  }
  if (!hasIndexTypes) {
    for (Type resultType : funcType.getResults()) {
      if (resultType.isIndex()) {
        hasIndexTypes = true;
        break;
      }
    }
  }
  
  if (!hasIndexTypes) {
    return failure(); // Nothing to convert
  }

  // Convert function signature types
  SmallVector<Type> newInputTypes;
  for (Type inputType : funcType.getInputs()) {
    if (inputType.isIndex()) {
      newInputTypes.push_back(IntegerType::get(rewriter.getContext(), 32));
    } else {
      newInputTypes.push_back(inputType);
    }
  }
  
  SmallVector<Type> newResultTypes;
  for (Type resultType : funcType.getResults()) {
    if (resultType.isIndex()) {
      newResultTypes.push_back(IntegerType::get(rewriter.getContext(), 32));
    } else {
      newResultTypes.push_back(resultType);
    }
  }
  
  // Create new function type
  auto newFuncType = FunctionType::get(rewriter.getContext(), newInputTypes, newResultTypes);
  
  // Update the function's type in place to avoid issues with region conversion
  funcOp.setType(newFuncType);
  
  // Update argument types in place
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    Value arg = funcOp.getArgument(i);
    if (arg.getType().isIndex()) {
      arg.setType(IntegerType::get(rewriter.getContext(), 32));
    }
  }
  
  return success();
}

} // namespace lowertocalyx
} // namespace circt