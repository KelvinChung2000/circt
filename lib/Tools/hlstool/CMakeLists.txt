add_circt_library(CIRCTHlsToolLib
  pipelineHLSFlow.cpp
  hlsPassPipelineHelper.cpp
  hlstoolOptions.cpp

  LINK_LIBS PUBLIC
  CIRCTSupport

  MLIRIR
  MLIRPass
  MLIRTransforms
  CIRCTAffineToLoopSchedule
  CIRCTLoopScheduleToCalyx
  CIRCTSCFToCalyx
)
