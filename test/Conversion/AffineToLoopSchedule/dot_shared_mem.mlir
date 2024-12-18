// RUN: circt-opt -convert-affine-to-loopschedule %s | FileCheck %s

// CHECK-LABEL: func @dot_shared_mem
func.func @dot_shared_mem(%arg0: memref<128xi32>) -> i32 {
  // LoopSchedule Pipeline boilerplate checked above, just check the stages computations.

  // CHECK: loopschedule.pipeline II = 2
  // First stage.
  // CHECK: %[[STAGE0:.+]]:3 = loopschedule.pipeline.stage
  // CHECK-DAG: %[[STAGE0_0:.+]] = memref.load %arg0[%arg1] : memref<128xi32>
  // CHECK-DAG: %[[STAGE0_1:.+]] = arith.addi %arg1, %c64 : index
  // CHECK-DAG: %[[STAGE0_2:.+]] = arith.addi %arg1, %c1 : index
  // CHECK: loopschedule.register %[[STAGE0_0]], %[[STAGE0_1]], %[[STAGE0_2]]

  // Second stage.
  // CHECK: %[[STAGE1:.+]]:2 = loopschedule.pipeline.stage
  // CHECK: %[[STAGE1_0:.+]] = memref.load %arg0[%[[STAGE0]]#1] : memref<128xi32>
  // CHECK: loopschedule.register %[[STAGE0]]#0, %[[STAGE1_0]]

  // Third stage.
  // CHECK: %[[STAGE2:.+]] = loopschedule.pipeline.stage
  // CHECK: %[[STAGE2_0:.+]] = arith.muli %[[STAGE1]]#0, %[[STAGE1]]#1 : i32
  // CHECK: loopschedule.register %[[STAGE2_0]]

  // Fourth stage.
  // CHECK: %[[STAGE3:.+]] = loopschedule.pipeline.stage
  // CHECK: %[[STAGE3_0:.+]] = arith.addi %arg2, %[[STAGE2]] : i32
  // CHECK: loopschedule.register %[[STAGE3_0]]

  // LoopSchedule Pipeline terminator.
  // CHECK: loopschedule.terminator iter_args(%[[STAGE0]]#2, %[[STAGE3]]), results(%[[STAGE3]])

  %c0_i32 = arith.constant 0 : i32
  %c64_index = arith.constant 64 : index
  %0 = affine.for %arg2 = 0 to 64 iter_args(%arg3 = %c0_i32) -> (i32) {
    %1 = affine.load %arg0[%arg2] : memref<128xi32>
    %2 = affine.load %arg0[%arg2 + %c64_index] : memref<128xi32>
    %3 = arith.muli %1, %2 : i32
    %4 = arith.addi %arg3, %3 : i32
    affine.yield %4 : i32
  }

  return %0 : i32
}
