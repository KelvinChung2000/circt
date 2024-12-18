// RUN: circt-opt -convert-affine-to-loopschedule %s | FileCheck %s

// CHECK-LABEL: func @perfectly_nested_loop
func.func @perfectly_nested_loop(%arg0 : memref<10x10xindex>) {
  // Outer Loop unchanged.
  // CHECK: affine.for %arg1 = 0 to 10

  // Setup constants.
  // CHECK: %[[LB:.+]] = arith.constant 0 : [[ITER_TYPE:.+]]
  // CHECK: %[[UB:.+]] = arith.constant [[TRIP_COUNT:.+]] : [[ITER_TYPE]]
  // CHECK: %[[STEP:.+]] = arith.constant 1 : [[ITER_TYPE]]

  // LoopSchedule Pipeline header.
  // CHECK: loopschedule.pipeline II = 1 trip_count = [[TRIP_COUNT]] iter_args(%[[ITER_ARG:.+]] = %[[LB]]) : ([[ITER_TYPE]]) -> ()

  // Condition block.
  // CHECK: %[[COND_RESULT:.+]] = arith.cmpi ult, %[[ITER_ARG]]
  // CHECK: loopschedule.register %[[COND_RESULT]]

  // First stage.
  // CHECK: %[[STAGE0:.+]] = loopschedule.pipeline.stage
  // CHECK: %[[ITER_INC:.+]] = arith.addi %[[ITER_ARG]], %[[STEP]]
  // CHECK: loopschedule.register %[[ITER_INC]]

  // LoopSchedule Pipeline terminator.
  // CHECK: loopschedule.terminator iter_args(%[[STAGE0]]), results()

  affine.for %arg1 = 0 to 10 {
    affine.for %arg2 = 0 to 10 {
      affine.store %arg1, %arg0[%arg1, %arg2] : memref<10x10xindex>
    }
  }

  return
}