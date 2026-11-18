% The kernel of a GPU's processes, merk_place, is the device compile's
% (docs/INPROGRESS.md, Register pressure in GPU processes): teams of 256
% threads, and the waves a SIMD that -fgpu-process-waves= gives, or else
% the GPU's default, which bound each lane's registers in all the code it
% calls. 0 leaves them to the backend. An image of a module with no kinds
% that run on a GPU has it too.
%
% REQUIRES: amdgpu-registered-target
% RUN: merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu9.0a-amd-amdhsa -emit-llvm %s -o - \
% RUN:   | FileCheck %s --check-prefixes=CHECK,W2
% RUN: merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - \
% RUN:   | FileCheck %s --check-prefixes=CHECK,NONE
% RUN: merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -fgpu-process-waves=8 -emit-llvm %s -o - \
% RUN:   | FileCheck %s --check-prefixes=CHECK,W8
% RUN: merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu9.0a-amd-amdhsa -fgpu-process-waves=0 -emit-llvm %s -o - \
% RUN:   | FileCheck %s --check-prefixes=CHECK,NONE
% RUN: not merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu9.0a-amd-amdhsa -fgpu-process-waves=33 -emit-llvm %s -o /dev/null 2>&1 \
% RUN:   | FileCheck %s --check-prefix=ERROR
% RUN: merkc -### -fprocess-declarations --offload-arch=gfx1030 -fgpu-process-waves=6 %s -o prog 2>&1 \
% RUN:   | FileCheck %s --check-prefix=DRIVER

% CHECK: define protected amdgpu_kernel void @merk_place(ptr %0) #[[ATTRS:[0-9]+]] {
% CHECK-NEXT: call void @merk_device_place(ptr %0)
% CHECK-NEXT: ret void
% CHECK: attributes #[[ATTRS]] = {
% CHECK-SAME: convergent
% CHECK-SAME: "amdgpu-flat-work-group-size"="256,256"
% W2-SAME: "amdgpu-waves-per-eu"="2"
% W8-SAME: "amdgpu-waves-per-eu"="8"
% NONE-NOT: "amdgpu-waves-per-eu"
% CHECK-SAME: }

% ERROR: error: -fgpu-process-waves=33: expected the waves a SIMD, from 0 to 32

% DRIVER: "-triple" "amdgpu10.30-amd-amdhsa"{{.*}} "-fprocess-kernels" "-fgpu-process-waves=6"

:- module process_place_kernel.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
main(!IO) :- io.write_string("hello\n", !IO).
