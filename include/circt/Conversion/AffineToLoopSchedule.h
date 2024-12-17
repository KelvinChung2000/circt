//===- AffineToLoopSchedule.h
//-------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_AFFINETOLOOPSCHEDULE_H_
#define CIRCT_CONVERSION_AFFINETOLOOPSCHEDULE_H_

#include "circt/Support/LLVM.h"
#include <memory>
#include <optional>
#include <string>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

#define GEN_PASS_DECL_AFFINETOLOOPSCHEDULE
#include "circt/Conversion/Passes.h.inc"


} // namespace circt

#endif // CIRCT_CONVERSION_AFFINETOLOOPSCHEDULE_H_
