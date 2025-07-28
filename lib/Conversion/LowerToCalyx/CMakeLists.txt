add_circt_conversion_library(CIRCTLowerToCalyx
  ArithToCalyx.cpp
  ControlFlowToCalyx.cpp
  FunctionToCalyx.cpp
  LowerToCalyx.cpp
  LowerToCalyxUtil.cpp
  MemrefToCalyx.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTCalyx
  CIRCTSupport
  CIRCTHW
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRTransforms
)