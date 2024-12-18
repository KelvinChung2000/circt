// RUN: hlstool --help | FileCheck %s --implicit-check-not='{{[Oo]}}ptions:'

// CHECK: OVERVIEW: CIRCT HLS tool
// CHECK: General {{[Oo]}}ptions
// CHECK: Generic Options
// CHECK: Pipeline Flow Options
// CHECK: hlstool Options
// CHECK: --lowering-options=
