#ifndef CIRCT_TOOLS_HLSTOOL_PIPELINEHLSFLOW_H
#define CIRCT_TOOLS_HLSTOOL_PIPELINEHLSFLOW_H

#include "circt/Tools/hlstool/hlstool.h"

namespace circt {
namespace hlstool {

class pipelineHLSFlow : HLSFlow {

public:
  pipelineHLSFlow(PassManager &pm, mlir::ModuleOp module)
      : HLSFlow(pm, module) {};

  ~pipelineHLSFlow() {};

  LogicalResult run();

  void preCompile() override;
  void core() override;
  void postCompile() override;
  void rtl() override;
  void sv() override;
};

void registerPipelineHLSCLOptions();
} // namespace hlstool
} // namespace circt

#endif // CIRCT_TOOLS_HLSTOOL_PIPELINEHLSFLOW_H