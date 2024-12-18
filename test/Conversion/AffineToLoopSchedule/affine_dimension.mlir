// RUN: circt-opt -convert-affine-to-loopschedule %s | FileCheck %s

// CHECK-LABEL: func @affine_dimension
#map1 = affine_map<(d0)[] -> (d0 + 1)>
func.func @affine_dimension(%arg0: i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[CAST:.+]] = arith.index_cast %arg0
  // CHECK-DAG: %[[UB:.+]] = arith.addi %[[CAST]], %[[C1]]
  %0 = arith.index_cast %arg0 : i32 to index
  // CHECK: arith.cmpi ult, %arg1, %[[UB]]
  %1 = affine.for %arg1 = 1 to #map1(%0) iter_args(%arg2 = %c0_i32) -> (i32) {
    %2 = arith.index_cast %arg1 : index to i32
    %3 = arith.addi %arg2, %2 : i32
    affine.yield %3 : i32
  }
  return %1 : i32
}
