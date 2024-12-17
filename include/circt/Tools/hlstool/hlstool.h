#ifndef CIRCT_TOOLS_HLSTOOL_HLSTOOL_H
#define CIRCT_TOOLS_HLSTOOL_HLSTOOL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

namespace circt {
namespace hlstool {

class HLSFlow {
public:
  HLSFlow(PassManager &pm, mlir::ModuleOp module) : pm(pm), module(module) {};

  virtual ~HLSFlow() {};

  virtual void preCompile() = 0;
  virtual void core() = 0;
  virtual void postCompile() = 0;
  virtual void rtl() = 0;
  virtual void sv() = 0;

protected:
  PassManager &pm;
  mlir::ModuleOp module;
};

enum IRLevel {
  // A high-level dialect like affine or scf
  High,
  // The IR right before the core lowering dialect
  PreCompile,
  // The IR in core dialect
  Core,
  // The lowest form of core IR (i.e. after all passes have run)
  PostCompile,
  // The IR after lowering is performed
  RTL,
  // System verilog representation
  SV
};

enum HLSFlowOptions {
  // Compilation through the dynamically scheduled handshake dialect.
  HLSFlowDynamicHW,
  // Compilation through Calyx's CIRCT lowering implementation.
  HLSFlowCalyxHW,
  // Compilation though the pipeline dialect.
  PipelineHLSFlow,
};

enum OutputFormatKind { OutputIR, OutputVerilog, OutputSplitVerilog };

class hlsFlags {
public:
  hlsFlags();

  std::string kernel;
};

// command line options
void registerHLSToolOptions();

// hls pass pipeline helpers
std::unique_ptr<Pass> createSimpleCanonicalizerPass();
void loadHWLoweringPipeline(OpPassManager &pm);

} // namespace hlstool

} // namespace circt

#endif // CIRCT_TOOLS_HLSTOOL_HLSTOOL_H