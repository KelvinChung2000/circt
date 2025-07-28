#ifndef CIRCT_CONVERSION_LOWERTOCALYX_LOWERTOCALYXUTIL_H
#define CIRCT_CONVERSION_LOWERTOCALYX_LOWERTOCALYXUTIL_H

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Region.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace lowertocalyx {

/// Helper function to safely move a region's content to another region
/// This handles the common case of moving blocks between operations
void moveRegionContents(mlir::Region &sourceRegion, mlir::Region &targetRegion,
                        mlir::ConversionPatternRewriter &rewriter);

// Creates a DictionaryAttr containing a unit attribute 'name'. Used for
// defining mandatory port attributes for calyx::ComponentOp's.
mlir::DictionaryAttr getMandatoryPortAttr(mlir::MLIRContext *ctx,
                                          llvm::StringRef name);

// Adds the mandatory Calyx component I/O ports (->[clk, reset, go], [done]->)
// to ports.
void addMandatoryComponentPorts(
    mlir::PatternRewriter &rewriter,
    llvm::SmallVectorImpl<circt::calyx::PortInfo> &ports);

std::string getOpUniqueName(mlir::Operation *op);

/// Resolves the done signal for a given value based on its producer operation.
/// This function analyzes the producer and returns the appropriate done signal:
/// - For calyx.register: returns register.done()
/// - For calyx.seq_mem: returns memory.done()
/// - For operations with multiple i1 results: finds and returns done signal
/// - For operations with calyx.done attribute: returns that result
/// - Fallback: creates and returns constant 1 for immediate values
mlir::Value resolveDoneSignalForValue(mlir::Value val, mlir::Location loc,
                                     mlir::OpBuilder &builder);

/// Overload that uses constant deduplication for component operations
mlir::Value resolveDoneSignalForValue(mlir::Value val, mlir::Location loc,
                                     mlir::OpBuilder &builder,
                                     mlir::Operation *componentOp);

/// Updates the component's done port connection with the provided done signal.
/// This function finds the component's done port and creates an assignment
/// to connect it to the new done signal, replacing any existing connection.
void updateComponentDoneConnection(circt::calyx::ComponentOp comp,
                                   mlir::Value newDone, mlir::Location loc,
                                   mlir::OpBuilder &builder);

/// Returns or creates a hw::ConstantOp with the specified value within the given operation.
/// If a constant with the same value and bit width already exists in the operation's body, returns that constant.
/// Otherwise, creates a new constant at the operation level and returns it.
/// If bitWidth is not provided (std::nullopt), defaults to 1 bit (suitable for boolean constants).
/// This helps avoid duplicate constant operations in the IR and ensures consistent placement.
/// Accepts either func::FuncOp or calyx::ComponentOp.
mlir::Value getOrCreateConstant(mlir::Operation *parentOp,
                               mlir::ConversionPatternRewriter &rewriter,
                               int64_t value, 
                               std::optional<unsigned> bitWidth = std::nullopt);

/// Overload for regular OpBuilder (for use in non-conversion contexts)
mlir::Value getOrCreateConstant(mlir::Operation *parentOp,
                               mlir::OpBuilder &builder,
                               int64_t value, 
                               std::optional<unsigned> bitWidth = std::nullopt);

/// Generic "find or create" utility for any operation type
/// This template function searches for existing operations of type OpType with matching attributes
/// and creates a new one if none exists, avoiding duplicates
template<typename OpType>
mlir::Operation* getOrCreateOperation(mlir::Operation *parentOp,
                                     mlir::OpBuilder &builder,
                                     mlir::Location loc,
                                     llvm::function_ref<bool(OpType)> matcher,
                                     llvm::function_ref<OpType()> creator) {
  // Get the body block from either func::FuncOp or calyx::ComponentOp
  mlir::Block *bodyBlock = nullptr;
  
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
  
  // Search for existing operation that matches the criteria
  for (auto &op : bodyBlock->getOperations()) {
    if (auto typedOp = dyn_cast<OpType>(op)) {
      if (matcher(typedOp)) {
        return typedOp.getOperation();
      }
    }
  }
  
  // If no matching operation exists, create a new one
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(bodyBlock);
  
  auto newOp = creator();
  return newOp.getOperation();
}

} // namespace lowertocalyx
} // namespace circt

#endif // CIRCT_CONVERSION_LOWERTOCALYX_LOWERTOCALYXUTIL_H