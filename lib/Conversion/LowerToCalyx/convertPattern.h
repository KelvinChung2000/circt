#ifndef CIRCT_CONVERSION_LOWERTOCALYX_CONVERTPATTERN_H
#define CIRCT_CONVERSION_LOWERTOCALYX_CONVERTPATTERN_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

// Forward declarations for Calyx lowering types
namespace circt {
namespace calyx {
class CalyxLoweringState;
class ComponentOp;
using PatternApplicationState =
    mlir::DenseMap<const mlir::RewritePattern *,
                   mlir::SmallPtrSet<mlir::Operation *, 16>>;
} // namespace calyx
} // namespace circt

namespace circt {
namespace lowertocalyx {
using namespace mlir;

// Helper function to get operation location as unique string
std::string getOpUniqueName(mlir::Operation *op);

// SCF patterns
template <typename OpType>
struct ScfToCalyxPattern : mlir::OpConversionPattern<OpType> {
  using mlir::OpConversionPattern<OpType>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

using ScfForToCalyxIfPattern = ScfToCalyxPattern<mlir::scf::ForOp>;
using ScfWhileToCalyxIfPattern = ScfToCalyxPattern<mlir::scf::WhileOp>;
using ScfYieldToCalyxIfPattern = ScfToCalyxPattern<mlir::scf::YieldOp>;
using ScfExecuteRegionToCalyxPattern =
    ScfToCalyxPattern<mlir::scf::ExecuteRegionOp>;
using ScfForallToCalyxPattern = ScfToCalyxPattern<mlir::scf::ForallOp>;
using ScfIndexSwitchToCalyxPattern =
    ScfToCalyxPattern<mlir::scf::IndexSwitchOp>;
using ScfParallelToCalyxPattern = ScfToCalyxPattern<mlir::scf::ParallelOp>;
using ScfConditionToCalyxPattern = ScfToCalyxPattern<mlir::scf::ConditionOp>;

// Arith patterns

template <typename SourceType, typename TargetType>
struct ArithBinaryOpToCalyxPattern : mlir::OpConversionPattern<SourceType> {
  using mlir::OpConversionPattern<SourceType>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(SourceType op, typename SourceType::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

template <typename SourceType, typename TargetType>
struct ArithPipelinedBinaryOpToCalyxPattern
    : mlir::OpConversionPattern<SourceType> {
  using mlir::OpConversionPattern<SourceType>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(SourceType op, typename SourceType::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

template <typename SourceType, typename TargetType>
struct ArithUnaryOpToCalyxPattern : mlir::OpConversionPattern<SourceType> {
  using mlir::OpConversionPattern<SourceType>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(SourceType op, typename SourceType::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

template <typename SourceType, typename TargetType>
struct ArithComparisonOpToCalyxPattern : mlir::OpConversionPattern<SourceType> {
  using mlir::OpConversionPattern<SourceType>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(SourceType op, typename SourceType::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

template <typename SourceType>
struct ArithSpecialOpToCalyxPattern : mlir::OpConversionPattern<SourceType> {
  using mlir::OpConversionPattern<SourceType>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(SourceType op, typename SourceType::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

// Binary integer arithmetic operations (non-pipelined, no done signal)
using ArithAddIToCalyxPattern =
    ArithBinaryOpToCalyxPattern<mlir::arith::AddIOp, calyx::AddLibOp>;
using ArithSubIToCalyxPattern =
    ArithBinaryOpToCalyxPattern<mlir::arith::SubIOp, calyx::SubLibOp>;

// Pipelined arithmetic operations (have done signal)
using ArithMulIToCalyxPattern =
    ArithPipelinedBinaryOpToCalyxPattern<mlir::arith::MulIOp,
                                         calyx::MultPipeLibOp>;
using ArithDivSIToCalyxPattern =
    ArithPipelinedBinaryOpToCalyxPattern<mlir::arith::DivSIOp,
                                         calyx::DivSPipeLibOp>;
using ArithDivUIToCalyxPattern =
    ArithPipelinedBinaryOpToCalyxPattern<mlir::arith::DivUIOp,
                                         calyx::DivUPipeLibOp>;
using ArithRemSIToCalyxPattern =
    ArithPipelinedBinaryOpToCalyxPattern<mlir::arith::RemSIOp,
                                         calyx::RemSPipeLibOp>;
using ArithRemUIToCalyxPattern =
    ArithPipelinedBinaryOpToCalyxPattern<mlir::arith::RemUIOp,
                                         calyx::RemUPipeLibOp>;

// Binary bitwise operations
using ArithAndIToCalyxPattern =
    ArithBinaryOpToCalyxPattern<mlir::arith::AndIOp, calyx::AndLibOp>;
using ArithOrIToCalyxPattern =
    ArithBinaryOpToCalyxPattern<mlir::arith::OrIOp, calyx::OrLibOp>;
using ArithXOrIToCalyxPattern =
    ArithBinaryOpToCalyxPattern<mlir::arith::XOrIOp, calyx::XorLibOp>;
using ArithShRSIToCalyxPattern =
    ArithBinaryOpToCalyxPattern<mlir::arith::ShRSIOp, calyx::SrshLibOp>;
using ArithShRUIToCalyxPattern =
    ArithBinaryOpToCalyxPattern<mlir::arith::ShRUIOp, calyx::ShruLibOp>;

// Binary floating-point operations
using ArithAddFToCalyxPattern =
    ArithBinaryOpToCalyxPattern<mlir::arith::AddFOp, calyx::AddFOpIEEE754>;
using ArithMulFToCalyxPattern =
    ArithBinaryOpToCalyxPattern<mlir::arith::MulFOp, calyx::MulFOpIEEE754>;

// Unary type conversion operations
using ArithExtSIToCalyxPattern =
    ArithUnaryOpToCalyxPattern<mlir::arith::ExtSIOp, calyx::ExtSILibOp>;
using ArithTruncIToCalyxPattern =
    ArithUnaryOpToCalyxPattern<mlir::arith::TruncIOp, calyx::SliceLibOp>;

// Comparison operations - TODO: Need proper Calyx comparison operations
using ArithCmpIToCalyxPattern =
    ArithComparisonOpToCalyxPattern<mlir::arith::CmpIOp, calyx::EqLibOp>;

// Special operations
using ArithConstantToCalyxPattern =
    ArithSpecialOpToCalyxPattern<mlir::arith::ConstantOp>;
using ArithSelectToCalyxPattern =
    ArithSpecialOpToCalyxPattern<mlir::arith::SelectOp>;
// Dedicated IndexCast pattern class
struct ArithIndexCastToCalyxPattern : mlir::OpConversionPattern<mlir::arith::IndexCastOp> {
  using mlir::OpConversionPattern<mlir::arith::IndexCastOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::IndexCastOp op, mlir::arith::IndexCastOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

// Function patterns
struct FuncFuncToCalyxPattern : mlir::OpConversionPattern<mlir::func::FuncOp> {
  using mlir::OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp op, mlir::func::FuncOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

struct FuncReturnToCalyxPattern
    : mlir::OpConversionPattern<mlir::func::ReturnOp> {
  using mlir::OpConversionPattern<mlir::func::ReturnOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::func::ReturnOp op, mlir::func::ReturnOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

struct FuncCallToCalyxPattern : mlir::OpConversionPattern<mlir::func::CallOp> {
  using mlir::OpConversionPattern<mlir::func::CallOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op, mlir::func::CallOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

// Complete function to component conversion pattern (final step)
struct CompleteFuncToComponentPattern
    : mlir::OpConversionPattern<mlir::func::FuncOp> {
  using mlir::OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp op, mlir::func::FuncOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

// Memory patterns

template <typename OpType>
struct MemoryAllocToCalyxPattern : mlir::OpConversionPattern<OpType> {
  using mlir::OpConversionPattern<OpType>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

struct MemoryLoadToCalyxPattern
    : mlir::OpConversionPattern<mlir::memref::LoadOp> {
  using mlir::OpConversionPattern<mlir::memref::LoadOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::memref::LoadOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

struct MemoryStoreToCalyxPattern
    : mlir::OpConversionPattern<mlir::memref::StoreOp> {
  using mlir::OpConversionPattern<mlir::memref::StoreOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::memref::StoreOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

// Memory allocation patterns
using MemoryAllocOpToCalyxPattern =
    MemoryAllocToCalyxPattern<mlir::memref::AllocOp>;
using MemoryAllocaOpToCalyxPattern =
    MemoryAllocToCalyxPattern<mlir::memref::AllocaOp>;

// Forward declaration for new memory pattern registration
void populateMemrefToCalyxPatterns(
    circt::calyx::CalyxLoweringState &loweringState,
    mlir::RewritePatternSet &patterns, mlir::LogicalResult &resRef,
    circt::calyx::PatternApplicationState &patternState,
    mlir::DenseMap<mlir::func::FuncOp, circt::calyx::ComponentOp> &map);

/// Complete memref.load to Calyx conversion pattern
class MemrefLoadToCalyxPattern : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, memref::LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Complete memref.store to Calyx conversion pattern
class MemrefStoreToCalyxPattern : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp storeOp, memref::StoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Complete memref.alloca to Calyx conversion pattern
class MemrefAllocaToCalyxPattern
    : public OpConversionPattern<memref::AllocaOp> {
public:
  using OpConversionPattern<memref::AllocaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocaOp allocaOp, memref::AllocaOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Complete function argument memref preparation pattern
class MemrefFunctionArgPattern
    : public OpConversionPattern<calyx::ComponentOp> {
public:
  using OpConversionPattern<calyx::ComponentOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(calyx::ComponentOp componentOp,
                  calyx::ComponentOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

// Index type conversion patterns
/// Function signature index conversion pattern
class FuncOpIndexConversionPattern : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, func::FuncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

// Control flow wrapping patterns
/// Control flow wrapping pattern to wrap standalone enable operations
class ControlFlowWrappingPattern
    : public OpConversionPattern<calyx::ControlOp> {
public:
  using OpConversionPattern<calyx::ControlOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(calyx::ControlOp controlOp, calyx::ControlOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to remove empty groups (those with only done signals) and their
/// enables
class EmptyGroupOptimizationPattern
    : public OpConversionPattern<calyx::ComponentOp> {
public:
  using OpConversionPattern<calyx::ComponentOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(calyx::ComponentOp componentOp,
                  calyx::ComponentOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace lowertocalyx
} // namespace circt

#endif // CIRCT_CONVERSION_LOWERTOCALYX_CONTROLFLOWCALYX_H