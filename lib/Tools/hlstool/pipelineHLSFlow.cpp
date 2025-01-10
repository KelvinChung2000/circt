#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <fstream>

#include "llvm/Support/Debug.h"

#include "circt/Conversion/AffineToLoopSchedule.h"
#include "circt/Conversion/CalyxToFSM.h"
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/Passes.h"
#include "circt/Conversion/SCFToCalyx.h"
#include "circt/Dialect/Calyx/CalyxDialect.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/Version.h"
#include "circt/Tools/hlstool/hlstool.h"
#include "circt/Tools/hlstool/pipelineHLSFlow.h"
#include "circt/Transforms/Passes.h"

#define DEBUG_TYPE "pipelineHLSFlow"
using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace circt::hlstool;

namespace {

static cl::OptionCategory pipelineFlowCategory("Pipeline Flow Options");
struct pipelineHLSFlowOptions {
  cl::opt<std::string> pipelineModeOpt{
      "pipeline-mode", cl::desc("Pipeline mode: full, inner-most"),
      cl::init("inner-most"), cl::cat(pipelineFlowCategory)};

  cl::opt<std::string> opCycleMapFile{
      "op-cycle-map",
      cl::desc("A directory that contains the operation cycle information"),
      cl::init(""), cl::cat(pipelineFlowCategory)};
};

} // namespace
static ManagedStatic<pipelineHLSFlowOptions> clOpts;
void hlstool::registerPipelineHLSCLOptions() { *clOpts; }

std::map<std::string, int>
pipelineHLSFlow::parseCycleOpMap(std::string opCycleMapFile) {
  std::map<std::string, int> cycleMap;
  if (opCycleMapFile.empty())
    return cycleMap;

  // Parse the cycle map file
  std::ifstream cycleMapFile(opCycleMapFile);
  if (!cycleMapFile.is_open()) {
    llvm::errs() << "Failed to open the cycle map file: " << opCycleMapFile;
    exit(1);
  }

  std::string line;
  while (std::getline(cycleMapFile, line)) {
    std::istringstream iss(line);
    std::string opName;
    int cycle;
    if (std::getline(iss, opName, ':') && (iss >> cycle)) {
      cycleMap[opName] = cycle;
      LLVM_DEBUG(llvm::dbgs()
                 << "Operation: " << opName << " cycles: " << cycle << "\n");
    } else {
      llvm::errs() << "Failed to parse line: " << line << "\n";
      exit(1);
    }
  }

  return cycleMap;
}

void pipelineHLSFlow::preCompile() {
  auto hlsflags = hlsFlags();
  pm.addPass(circt::createFlattenMemRefPass());
  pm.nest<func::FuncOp>().addPass(circt::createAffineToLoopSchedule(
      {clOpts->pipelineModeOpt, parseCycleOpMap(clOpts->opCycleMapFile)}));
  pm.addPass(mlir::createLowerAffinePass());
  // pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createForToWhileLoopPass());
}

void pipelineHLSFlow::core() {
  // pm.addPass(circt::createSCFToCalyxPass());
  pm.addPass(circt::createLoopScheduleToCalyxPass());
}

void pipelineHLSFlow::postCompile() {
  pm.addPass(createSimpleCanonicalizerPass());
  // Eliminate Calyx's comb group abstraction
  pm.addNestedPass<calyx::ComponentOp>(
      circt::calyx::createRemoveCombGroupsPass());
  pm.addPass(createSimpleCanonicalizerPass());

  // Compile to FSM
  pm.addNestedPass<calyx::ComponentOp>(circt::createCalyxToFSMPass());
  pm.addPass(createSimpleCanonicalizerPass());
  pm.addNestedPass<calyx::ComponentOp>(
      circt::createMaterializeCalyxToFSMPass());
  pm.addPass(createSimpleCanonicalizerPass());

  // Eliminate Calyx's group abstraction
  pm.addNestedPass<calyx::ComponentOp>(circt::createRemoveGroupsFromFSMPass());
  pm.addPass(createSimpleCanonicalizerPass());
}

void pipelineHLSFlow::rtl() {
  pm.addPass(circt::createCalyxToHWPass());
  pm.addPass(createSimpleCanonicalizerPass());
}

void pipelineHLSFlow::sv() {
  pm.addPass(circt::createConvertFSMToSVPass());
  pm.addPass(createSimpleCanonicalizerPass());
  pm.nest<hw::HWModuleOp>().addPass(circt::seq::createLowerSeqHLMemPass());
  pm.addPass(seq::createHWMemSimImplPass());
  pm.addPass(circt::createLowerSeqToSVPass());
  pm.nest<hw::HWModuleOp>().addPass(sv::createHWCleanupPass());

  // Legalize unsupported operations within the modules.
  pm.nest<hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());
  pm.addPass(createSimpleCanonicalizerPass());

  // Tidy up the IR to improve verilog emission quality.
  auto &modulePM = pm.nest<hw::HWModuleOp>();
  modulePM.addPass(sv::createPrettifyVerilogPass());
}
