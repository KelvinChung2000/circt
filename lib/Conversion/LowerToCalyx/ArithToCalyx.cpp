//===- ArithToCalyx.cpp - Arith to Calyx Conversion --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the template implementations for converting arith
// operations to Calyx.
//
//===----------------------------------------------------------------------===//

#include "LowerToCalyxUtil.h"
#include "circt/Dialect/Calyx/CalyxLoweringUtils.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "convertPattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

// Simple container for comparison lib op ports
struct ComparisonPorts {
  Value leftPort;
  Value rightPort;
  Value outPort;
};

// Create the appropriate Calyx comparison library operation based on the
// arith integer comparison predicate and return its key ports.
static ComparisonPorts createComparisonLibOp(arith::CmpIPredicate pred,
                                             OpBuilder &builder, Location loc,
                                             StringRef symName,
                                             ArrayRef<Type> resultTypes) {
  switch (pred) {
  case arith::CmpIPredicate::eq: {
    auto op = builder.create<calyx::EqLibOp>(
        loc, builder.getStringAttr(symName), resultTypes);
    return {op.getLeft(), op.getRight(), op.getOut()};
  }
  case arith::CmpIPredicate::ne: {
    auto op = builder.create<calyx::NeqLibOp>(
        loc, builder.getStringAttr(symName), resultTypes);
    return {op.getLeft(), op.getRight(), op.getOut()};
  }

  // Signed integer comparisons
  case arith::CmpIPredicate::slt: {
    auto op = builder.create<calyx::SltLibOp>(
        loc, builder.getStringAttr(symName), resultTypes);
    return {op.getLeft(), op.getRight(), op.getOut()};
  }
  case arith::CmpIPredicate::sle: {
    auto op = builder.create<calyx::SleLibOp>(
        loc, builder.getStringAttr(symName), resultTypes);
    return {op.getLeft(), op.getRight(), op.getOut()};
  }
  case arith::CmpIPredicate::sgt: {
    auto op = builder.create<calyx::SgtLibOp>(
        loc, builder.getStringAttr(symName), resultTypes);
    return {op.getLeft(), op.getRight(), op.getOut()};
  }
  case arith::CmpIPredicate::sge: {
    auto op = builder.create<calyx::SgeLibOp>(
        loc, builder.getStringAttr(symName), resultTypes);
    return {op.getLeft(), op.getRight(), op.getOut()};
  }

  // Unsigned integer comparisons
  case arith::CmpIPredicate::ult: {
    auto op = builder.create<calyx::LtLibOp>(
        loc, builder.getStringAttr(symName), resultTypes);
    return {op.getLeft(), op.getRight(), op.getOut()};
  }
  case arith::CmpIPredicate::ule: {
    auto op = builder.create<calyx::LeLibOp>(
        loc, builder.getStringAttr(symName), resultTypes);
    return {op.getLeft(), op.getRight(), op.getOut()};
  }
  case arith::CmpIPredicate::ugt: {
    auto op = builder.create<calyx::GtLibOp>(
        loc, builder.getStringAttr(symName), resultTypes);
    return {op.getLeft(), op.getRight(), op.getOut()};
  }
  case arith::CmpIPredicate::uge: {
    auto op = builder.create<calyx::GeLibOp>(
        loc, builder.getStringAttr(symName), resultTypes);
    return {op.getLeft(), op.getRight(), op.getOut()};
  }
  }

  llvm_unreachable("Unhandled arith::CmpIPredicate in createComparisonLibOp");
}

// Template function implementation for binary operations
template <typename SourceType, typename TargetType>
LogicalResult
ArithBinaryOpToCalyxPattern<SourceType, TargetType>::matchAndRewrite(
    SourceType op, typename SourceType::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  static_assert(TargetType::template hasTrait<calyx::BinaryOpTrait>(),
                "TargetType must have BinaryOpTrait");

  // Get operation location and operands
  auto loc = op.getLoc();
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();

  // Get the result type width
  // Canonicalize result type: apply TypeConverter (if any), map index -> i32,
  // and obtain an IntegerType. Implemented as a small lambda to avoid
  // duplicating this logic.
  auto normalizeToIntegerType = [&](Type t) -> IntegerType {
    Type ty = t;
    if (auto *tc = this->getTypeConverter())
      if (Type converted = tc->convertType(ty))
        ty = converted;
    if (isa<IndexType>(ty))
      ty = IntegerType::get(op.getContext(), 32);
    return dyn_cast<IntegerType>(ty);
  };
  Type resultType = op.getResult().getType();
  auto intType = normalizeToIntegerType(resultType);
  if (!intType)
    return rewriter.notifyMatchFailure(op, "Only integer types supported");

  // Create a simple symbol name using the general utility function
  std::string symName = getOpUniqueName(op);

  // Create result types: left input, right input, output (all same width for
  // binary ops)
  SmallVector<Type> resultTypes = {resultType, resultType, resultType};

  // Find the parent component to create library operations at the component
  // level
  calyx::ComponentOp topLevelOp =
      op->template getParentOfType<calyx::ComponentOp>();
  auto wiresOp = topLevelOp.getWiresOp();

  // Create the library operation inside the wires operation
  OpBuilder componentBuilder(wiresOp);

  // Create the appropriate Calyx library operation based on the arith operation
  // type
  TargetType libOp = componentBuilder.create<TargetType>(
      loc, componentBuilder.getStringAttr(symName), resultTypes);

  // Get the ports of the library operation using proper accessors
  Value leftPort = libOp.getLeft();   // left input port
  Value rightPort = libOp.getRight(); // right input port
  Value outPort = libOp.getOut();     // output port

  // Create assign operations in the wires section (not in groups)
  // Use the same wiresBlock we created the library operation in
  auto &wiresBlock = wiresOp.getBodyRegion().front();
  OpBuilder wiresBuilder(&wiresBlock, wiresBlock.end());

  wiresBuilder.create<calyx::AssignOp>(loc, leftPort, lhs);
  wiresBuilder.create<calyx::AssignOp>(loc, rightPort, rhs);

  // Replace the original arith operation result with the library operation
  // output
  rewriter.replaceOp(op, outPort);

  return success();
}

// Template function implementation for pipelined binary operations
template <typename SourceType, typename TargetType>
LogicalResult
ArithPipelinedBinaryOpToCalyxPattern<SourceType, TargetType>::matchAndRewrite(
    SourceType op, typename SourceType::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  static_assert(TargetType::template hasTrait<calyx::PipelineTrait>(),
                "TargetType must have PipelineTrait");
  static_assert(TargetType::template hasTrait<calyx::BinaryOpTrait>(),
                "TargetType must have BinaryOpTrait");

  // Get operation location and operands
  auto loc = op.getLoc();
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();

  // Get the result type width
  // Canonicalize result type (lambda as above).
  auto normalizeToIntegerType = [&](Type t) -> IntegerType {
    Type ty = t;
    if (auto *tc = this->getTypeConverter())
      if (Type converted = tc->convertType(ty))
        ty = converted;
    if (isa<IndexType>(ty))
      ty = IntegerType::get(op.getContext(), 32);
    return dyn_cast<IntegerType>(ty);
  };
  Type resultType = op.getResult().getType();
  auto intType = normalizeToIntegerType(resultType);
  if (!intType)
    return rewriter.notifyMatchFailure(op, "Only integer types supported");

  // Width may be used later for pipelined operations

  // Create a simple symbol name using the general utility function
  std::string symName = getOpUniqueName(op);

  // Create result types for pipelined operations: clk, reset, go, left, right,
  // out, done
  Type i1Type = rewriter.getI1Type();
  SmallVector<Type> resultTypes = {i1Type,     i1Type,     i1Type, resultType,
                                   resultType, resultType, i1Type};

  auto topLevelOp = op->template getParentOfType<calyx::ComponentOp>();
  auto wiresOp = topLevelOp.getWiresOp();

  OpBuilder componentBuilder(rewriter.getContext());
  componentBuilder.setInsertionPoint(wiresOp);

  // Create the appropriate Calyx pipelined library operation
  TargetType libOp = componentBuilder.create<TargetType>(
      loc, componentBuilder.getStringAttr(symName), resultTypes);

  // Get the ports of the library operation (7 results: clk, reset, go, left,
  // right, out, done)
  Value leftPort = libOp.getLeft();   // left input port
  Value rightPort = libOp.getRight(); // right input port
  Value outPort = libOp.getOut();     // output port

  // Create assign operations in the wires section
  auto &wiresBlock = wiresOp.getBodyRegion().front();
  OpBuilder wiresBuilder(&wiresBlock, wiresBlock.end());

  wiresBuilder.create<calyx::AssignOp>(loc, leftPort, lhs);
  wiresBuilder.create<calyx::AssignOp>(loc, rightPort, rhs);

  // Resolve the go signal based on the done signals of the operands
  Value leftDone = resolveDoneSignalForValue(lhs, topLevelOp);
  Value rightDone = resolveDoneSignalForValue(rhs, topLevelOp);
  Value goSignal = createAndGate(leftDone, rightDone, op, topLevelOp);

  rewriter.create<calyx::AssignOp>(loc, libOp.getGo(), goSignal);

  // Replace the original arith operation result with the library operation
  // output
  rewriter.replaceOp(op, outPort);

  return success();
}

// Template function implementation for unary operations
template <typename SourceType, typename TargetType>
LogicalResult
ArithUnaryOpToCalyxPattern<SourceType, TargetType>::matchAndRewrite(
    SourceType op, typename SourceType::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Get operation location and operand
  auto loc = op.getLoc();
  Value operand = adaptor.getIn();

  // Get the input and result types
  auto inputType = operand.getType();
  auto resultType = op.getResult().getType();

  auto inputIntType = dyn_cast<IntegerType>(inputType);
  auto resultIntType = dyn_cast<IntegerType>(resultType);
  if (!inputIntType || !resultIntType) {
    return rewriter.notifyMatchFailure(op, "Only integer types supported");
  }

  // Create a simple symbol name using the general utility function
  std::string symName = getOpUniqueName(op);

  // Find the parent component to create library operations at the component
  // level
  auto componentOp = op->template getParentOfType<calyx::ComponentOp>();
  if (!componentOp) {
    return rewriter.notifyMatchFailure(
        op, "Unary operation not within a calyx.component");
  }

  // Create the library operation at the component level (before wires section)
  auto wiresOp =
      *componentOp.getBodyBlock()->template getOps<calyx::WiresOp>().begin();
  OpBuilder componentBuilder(rewriter.getContext());
  componentBuilder.setInsertionPoint(wiresOp);

  // For truncation (SliceLibOp), we need input, output types and slice
  // parameters
  if constexpr (std::is_same_v<TargetType, calyx::SliceLibOp>) {
    // SliceLibOp parameters: input width, output width
    SmallVector<Type> resultTypes = {inputType, resultType};

    // Create slice operation with parameters
    auto sliceOp = componentBuilder.create<TargetType>(
        loc, componentBuilder.getStringAttr(symName), resultTypes);

    // Get the ports of the library operation
    Value inputPort = sliceOp->getResult(0); // input port
    Value outPort = sliceOp->getResult(1);   // output port

    // Create assign operations in the wires section
    auto &wiresBlock = wiresOp.getBodyRegion().front();
    OpBuilder wiresBuilder(&wiresBlock, wiresBlock.end());

    wiresBuilder.create<calyx::AssignOp>(loc, inputPort, operand);

    // Replace the original operation result with the library operation output
    rewriter.replaceOp(op, outPort);
  } else {
    // For other unary operations like ExtSI
    SmallVector<Type> resultTypes = {inputType, resultType};

    Operation *libOp = componentBuilder.create<TargetType>(
        loc, componentBuilder.getStringAttr(symName), resultTypes);

    // Get the ports of the library operation
    Value inputPort = libOp->getResult(0); // input port
    Value outPort = libOp->getResult(1);   // output port

    // Create assign operations in the wires section
    auto &wiresBlock = wiresOp.getBodyRegion().front();
    OpBuilder wiresBuilder(&wiresBlock, wiresBlock.end());

    wiresBuilder.create<calyx::AssignOp>(loc, inputPort, operand);

    // Replace the original operation result with the library operation output
    rewriter.replaceOp(op, outPort);
  }

  return success();
}

// Template function implementation for comparison operations
LogicalResult ArithCmpIToCalyxPattern::matchAndRewrite(
    arith::CmpIOp op, arith::CmpIOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Get operation location and operands
  auto loc = op.getLoc();
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();

  // Get the result type width
  // Canonicalize operand/result width using a lambda (apply TypeConverter,
  // map index -> i32).
  auto normalizeToIntegerType = [&](Type t) -> IntegerType {
    Type ty = t;
    if (auto *tc = this->getTypeConverter())
      if (Type converted = tc->convertType(ty))
        ty = converted;
    if (isa<IndexType>(ty))
      ty = IntegerType::get(op.getContext(), 32);
    return dyn_cast<IntegerType>(ty);
  };

  Type lResultType = normalizeToIntegerType(lhs.getType());
  Type rResultType = normalizeToIntegerType(rhs.getType());
  if (!lResultType || !rResultType)
    return rewriter.notifyMatchFailure(op, "Only integer types supported");

  // Create a simple symbol name using the general utility function
  std::string symName = getOpUniqueName(op);

  // Create result types: left input, right input, output (all same width for
  // binary ops)
  SmallVector<Type> resultTypes = {lResultType, rResultType,
                                   rewriter.getI1Type()};

  // Find the parent component to create library operations at the component
  // level
  calyx::ComponentOp topLevelOp =
      op->template getParentOfType<calyx::ComponentOp>();
  auto wiresOp = topLevelOp.getWiresOp();

  // Create the library operation inside the wires operation
  OpBuilder componentBuilder(wiresOp);

  // Create the appropriate Calyx library operation based on the predicate
  auto ports = createComparisonLibOp(op.getPredicate(), componentBuilder, loc,
                                     symName, resultTypes);

  // Use the ports from the helper function
  Value leftPort = ports.leftPort;
  Value rightPort = ports.rightPort;
  Value outPort = ports.outPort;

  // Create assign operations in the wires section (not in groups)
  // Use the same wiresBlock we created the library operation in
  auto &wiresBlock = wiresOp.getBodyRegion().front();
  OpBuilder wiresBuilder(&wiresBlock, wiresBlock.end());

  wiresBuilder.create<calyx::AssignOp>(loc, leftPort, lhs);
  wiresBuilder.create<calyx::AssignOp>(loc, rightPort, rhs);

  // Replace the original arith operation result with the library operation
  // output
  rewriter.replaceOp(op, outPort);

  return success();
}

LogicalResult ArithSelectToCalyxPattern::matchAndRewrite(
    arith::SelectOp op, arith::SelectOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Get operation location and operands
  auto loc = op.getLoc();
  Value lhs = adaptor.getTrueValue();
  Value rhs = adaptor.getFalseValue();
  Value cond = adaptor.getCondition();

  // Get the result type width
  // Canonicalize result type (lambda).
  auto normalizeToIntegerType = [&](Type t) -> IntegerType {
    Type ty = t;
    if (auto *tc = this->getTypeConverter())
      if (Type converted = tc->convertType(ty))
        ty = converted;
    if (isa<IndexType>(ty))
      ty = IntegerType::get(op.getContext(), 32);
    return dyn_cast<IntegerType>(ty);
  };
  Type resultType = op.getResult().getType();
  auto intType = normalizeToIntegerType(resultType);
  if (!intType)
    return rewriter.notifyMatchFailure(op, "Only integer types supported");

  // Create a simple symbol name using the general utility function
  std::string symName = getOpUniqueName(op);

  SmallVector<Type> resultTypes = {rewriter.getI1Type(), resultType, resultType,
                                   resultType};

  // Find the parent component to create library operations at the component
  // level
  calyx::ComponentOp topLevelOp =
      op->template getParentOfType<calyx::ComponentOp>();
  auto wiresOp = topLevelOp.getWiresOp();

  // Create the library operation inside the wires operation
  OpBuilder componentBuilder(wiresOp);
  auto muxOp = componentBuilder.create<calyx::MuxLibOp>(
      loc, componentBuilder.getStringAttr(symName), resultTypes);

  // Create assign operations in the wires section (not in groups)
  // Use the same wiresBlock we created the library operation in
  auto &wiresBlock = wiresOp.getBodyRegion().front();
  OpBuilder wiresBuilder(&wiresBlock, wiresBlock.end());

  wiresBuilder.create<calyx::AssignOp>(loc, muxOp.getTru(), lhs);
  wiresBuilder.create<calyx::AssignOp>(loc, muxOp.getFal(), rhs);
  wiresBuilder.create<calyx::AssignOp>(loc, muxOp.getCond(), cond);

  // Replace the original arith operation result with the library operation
  // output
  rewriter.replaceOp(op, muxOp.getOut());

  return success();
}

// Template function implementation for special operations
LogicalResult ArithConstantToCalyxPattern::matchAndRewrite(
    mlir::arith::ConstantOp op, mlir::arith::ConstantOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto loc = op.getLoc();
  auto value = op.getValue();
  auto type = op.getType();

  // Find the parent component to create constant at the component level
  auto componentOp = op->template getParentOfType<calyx::ComponentOp>();
  if (!componentOp) {
    return rewriter.notifyMatchFailure(
        op, "Constant operation not within a calyx.component");
  }

  // Create the constant at the component level (before wires section)
  auto wiresOp =
      *componentOp.getBodyBlock()->template getOps<calyx::WiresOp>().begin();
  OpBuilder componentBuilder(rewriter.getContext());
  componentBuilder.setInsertionPoint(wiresOp);

  // Handle constants following the same pattern as SCFToCalyx
  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    // Integer constants use hw::ConstantOp (as per SCFToCalyx)
    // Use getOrCreateConstant utility to deduplicate constants

    // Handle bit width calculation - index types should have been converted
    // to i32
    unsigned bitWidth;
    if (intAttr.getType().isIndex()) {
      // Index types are converted to i32 by the index conversion pass
      bitWidth = 32;
    } else {
      bitWidth = intAttr.getType().getIntOrFloatBitWidth();
    }

    auto constantValue = getOrCreateConstant(componentOp.getOperation(),
                                             intAttr.getInt(), bitWidth);
    rewriter.replaceOp(op, constantValue);
    return success();
  } else if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
    // Floating point constants use calyx::ConstantOp (as per SCFToCalyx)
    std::string constName = getOpUniqueName(op);
    auto calyxConst = componentBuilder.create<calyx::ConstantOp>(
        loc, componentBuilder.getStringAttr(constName), floatAttr, type);
    rewriter.replaceOp(op, calyxConst.getOut());
    return success();
  } else {
    return rewriter.notifyMatchFailure(
        op, "Unsupported constant type - only integer and float constants "
            "supported");
  }
}

// Explicit template instantiations for binary integer arithmetic operations
// (non-pipelined)
template struct ArithBinaryOpToCalyxPattern<mlir::arith::AddIOp,
                                            calyx::AddLibOp>;
template struct ArithBinaryOpToCalyxPattern<mlir::arith::SubIOp,
                                            calyx::SubLibOp>;

// Explicit template instantiations for pipelined binary integer arithmetic
// operations
template struct ArithPipelinedBinaryOpToCalyxPattern<mlir::arith::MulIOp,
                                                     calyx::MultPipeLibOp>;
template struct ArithPipelinedBinaryOpToCalyxPattern<mlir::arith::DivSIOp,
                                                     calyx::DivSPipeLibOp>;
template struct ArithPipelinedBinaryOpToCalyxPattern<mlir::arith::DivUIOp,
                                                     calyx::DivUPipeLibOp>;
template struct ArithPipelinedBinaryOpToCalyxPattern<mlir::arith::RemSIOp,
                                                     calyx::RemSPipeLibOp>;
template struct ArithPipelinedBinaryOpToCalyxPattern<mlir::arith::RemUIOp,
                                                     calyx::RemUPipeLibOp>;
// template struct ArithBinaryOpToCalyxPattern<mlir::arith::CeilDivSIOp,
//                                             calyx::CeilDivSPipeLibOp>;
// template struct ArithBinaryOpToCalyxPattern<mlir::arith::FloorDivSIOp,
//                                             calyx::FloorDivSPipeLibOp>;

// Explicit template instantiations for binary bitwise operations
template struct ArithBinaryOpToCalyxPattern<mlir::arith::AndIOp,
                                            calyx::AndLibOp>;
template struct ArithBinaryOpToCalyxPattern<mlir::arith::OrIOp, calyx::OrLibOp>;
template struct ArithBinaryOpToCalyxPattern<mlir::arith::XOrIOp,
                                            calyx::XorLibOp>;

template struct ArithBinaryOpToCalyxPattern<mlir::arith::ShRUIOp,
                                            calyx::RshLibOp>;
template struct ArithBinaryOpToCalyxPattern<mlir::arith::ShRSIOp,
                                            calyx::SrshLibOp>;
template struct ArithBinaryOpToCalyxPattern<mlir::arith::ShLIOp,
                                            calyx::LshLibOp>;

// Explicit template instantiations for binary floating-point operations
template struct ArithBinaryOpToCalyxPattern<mlir::arith::AddFOp,
                                            calyx::AddFOpIEEE754>;
// template struct ArithBinaryOpToCalyxPattern<mlir::arith::SubFOp,
// calyx::SubFOpIEEE754>;
template struct ArithBinaryOpToCalyxPattern<mlir::arith::MulFOp,
                                            calyx::MulFOpIEEE754>;
// template struct ArithBinaryOpToCalyxPattern<mlir::arith::DivFOp,
// calyx::Dive>;

// Explicit template instantiations for unary type conversion operations
template struct ArithUnaryOpToCalyxPattern<mlir::arith::ExtSIOp,
                                           calyx::ExtSILibOp>;
template struct ArithUnaryOpToCalyxPattern<mlir::arith::TruncIOp,
                                           calyx::SliceLibOp>;
// Note: Incomplete unary operations - commented out until proper Calyx target
// types are available template struct
// ArithUnaryOpToCalyxPattern<mlir::arith::ExtUIOp, calyx::ExtUILibOp>; template
// struct ArithUnaryOpToCalyxPattern<mlir::arith::TruncIOp, calyx::TruncLibOp>;
// template struct ArithUnaryOpToCalyxPattern<mlir::arith::IndexCastOp,
// calyx::IndexCastLibOp>; template struct
// ArithUnaryOpToCalyxPattern<mlir::arith::BitcastOp, calyx::BitcastLibOp>;

// // Explicit template instantiations for comparison operations
// template struct ArithComparisonOpToCalyxPattern<mlir::arith::CmpIOp,
//                                                 calyx::EqLibOp>;
// template struct ArithComparisonOpToCalyxPattern<mlir::arith::CmpFOp,
// calyx::CompareFOpIEEE754>;

// Dedicated IndexCast pattern implementation following SCFToCalyx approach
LogicalResult ArithIndexCastToCalyxPattern::matchAndRewrite(
    mlir::arith::IndexCastOp op, mlir::arith::IndexCastOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Normalize to integer types for bitwidth reasoning.
  Type sourceType = calyx::normalizeType(rewriter, op.getOperand().getType());
  Type targetType = calyx::normalizeType(rewriter, op.getResult().getType());
  unsigned targetBits = targetType.getIntOrFloatBitWidth();
  unsigned sourceBits = sourceType.getIntOrFloatBitWidth();

  auto loc = op.getLoc();
  Value inVal = adaptor.getIn();
  if (!inVal)
    return rewriter.notifyMatchFailure(op, "IndexCast adaptor operand is null");

  if (sourceBits == targetBits) {
    rewriter.replaceOp(op, inVal);
    return success();
  }

  // Re-express index_cast as ext/trunc arith op; existing patterns lower
  // those to PadLibOp / SliceLibOp.
  if (sourceBits < targetBits) {
    auto newType = IntegerType::get(op.getContext(), targetBits);
    auto ext = rewriter.create<arith::ExtSIOp>(loc, newType, inVal);
    rewriter.replaceOp(op, ext.getResult());
  } else {
    auto newType = IntegerType::get(op.getContext(), targetBits);
    auto trunc = rewriter.create<arith::TruncIOp>(loc, newType, inVal);
    rewriter.replaceOp(op, trunc.getResult());
  }
  return success();
}

} // namespace lowertocalyx
} // namespace circt