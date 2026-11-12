// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -O0 -triple amdgpu10.30-amd-amdhsa -emit-llvm -o - %s | FileCheck %s

typedef unsigned int uint;

// CHECK-LABEL: @test_callee_stack_size
// CHECK: call i32 @llvm.amdgcn.callee.stack.size()
void test_callee_stack_size(global uint* out) {
  *out = __builtin_amdgcn_callee_stack_size();
}
