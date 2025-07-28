//===- LowerToCalyx.h - Unified Lowering Pass Entry Point ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the unified LowerToCalyx
// pass constructor that can lower both SCF and LoopSchedule dialects to Calyx.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_LOWERTOCALYX_H
#define CIRCT_CONVERSION_LOWERTOCALYX_H

#include "circt/Support/LLVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <memory>

namespace circt {

#define GEN_PASS_DECL_LOWERTOCALYX
#include "circt/Conversion/Passes.h.inc"

namespace lowerToCalyx {
// If this attribute is set as a FuncOp argument or result attribute, it will be
// used as the Calyx port name.
static constexpr std::string_view sPortNameAttr = "calyx.port_name";

} // namespace lowerToCalyx

/// Create a unified lowering pass that can lower both SCF and LoopSchedule
/// dialects to Calyx.
std::unique_ptr<OperationPass<ModuleOp>>
createLowerToCalyxPass(std::string topLevelFunction = "");

} // namespace circt

#endif // CIRCT_CONVERSION_LOWERTOCALYX_H