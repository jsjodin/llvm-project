% The kernel of a GPU's processes, merk_place, is the device compile's
% (docs/INPROGRESS.md, Register pressure in GPU processes): teams of 256
% threads, and the waves a SIMD that -fgpu-process-waves= gives, which bound
% each lane's registers in all the code it calls. 0, or no option, leaves
% them to the backend. -fgpu-process-no-callee-saved gives it, and so every
% function that only it reaches, no callee-saved registers, and
% -fgpu-process-callee-saved keeps them, the last of the two given. The
% driver gives each GPU's defaults, as Clang's does to -cc1: on gfx90a 2 waves
% and no callee-saved registers, elsewhere neither.
% An image of a module with no kinds that run on a GPU has it too. The
% loops' procedures take their team-uniform parameters in SGPRs (inreg): all
% but the thread's start and its partial results.
%
% REQUIRES: amdgpu-registered-target
% RUN: merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu9.0a-amd-amdhsa -emit-llvm %s -o - \
% RUN:   | FileCheck %s --check-prefixes=CHECK,NONE,KEEPS
% RUN: merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu9.0a-amd-amdhsa -fgpu-process-waves=2 -fgpu-process-no-callee-saved -emit-llvm %s -o - \
% RUN:   | FileCheck %s --check-prefixes=CHECK,W2,NCS
% RUN: merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu9.0a-amd-amdhsa -fgpu-process-no-callee-saved -fgpu-process-callee-saved -emit-llvm %s -o - \
% RUN:   | FileCheck %s --check-prefixes=CHECK,NONE,KEEPS
% RUN: merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -fgpu-process-waves=8 -emit-llvm %s -o - \
% RUN:   | FileCheck %s --check-prefixes=CHECK,W8,KEEPS
% RUN: merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -fgpu-process-waves=8 -fgpu-process-no-callee-saved -emit-llvm %s -o - \
% RUN:   | FileCheck %s --check-prefixes=CHECK,W8,NCS
% RUN: merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu9.0a-amd-amdhsa -fgpu-process-waves=0 -emit-llvm %s -o - \
% RUN:   | FileCheck %s --check-prefixes=CHECK,NONE
% RUN: not merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu9.0a-amd-amdhsa -fgpu-process-waves=33 -emit-llvm %s -o /dev/null 2>&1 \
% RUN:   | FileCheck %s --check-prefix=ERROR
% RUN: merkc -### -fprocess-declarations --offload-arch=gfx1030 -fgpu-process-waves=6 -fgpu-process-no-callee-saved %s -o prog 2>&1 \
% RUN:   | FileCheck %s --check-prefix=DRIVER
% RUN: merkc -### -fprocess-declarations --offload-arch=gfx90a %s -o prog 2>&1 \
% RUN:   | FileCheck %s --check-prefix=DRIVER-GFX90A
% RUN: merkc -### -fprocess-declarations --offload-arch=gfx1030 %s -o prog 2>&1 \
% RUN:   | FileCheck %s --check-prefix=DRIVER-GFX1030
% RUN: merkc -### -fprocess-declarations --offload-arch=gfx90a -fgpu-process-waves=0 -fgpu-process-no-callee-saved -fgpu-process-callee-saved %s -o prog 2>&1 \
% RUN:   | FileCheck %s --check-prefix=DRIVER-OVERRIDE

% CHECK: define protected amdgpu_kernel void @merk_place(ptr %0) #[[ATTRS:[0-9]+]] {
% CHECK-NEXT: call void @merk_device_place(ptr %0)
% CHECK-NEXT: ret void
% CHECK: define hidden void @merk_process_loop(i64 inreg %0, i64 %1, i64 inreg %2, i64 inreg %3, i64 inreg %4, i64 inreg %5, i64 inreg %6, i64 inreg %7, i64 %8)
% CHECK: define hidden void @merk_team_loop(i64 inreg %0, i64 %1, i64 inreg %2, i64 inreg %3, i64 inreg %4, i64 inreg %5, i64 inreg %6, i64 inreg %7, i64 %8)
% CHECK: attributes #[[ATTRS]] = {
% CHECK-SAME: convergent
% CHECK-SAME: "amdgpu-flat-work-group-size"="256,256"
% W2-SAME: "amdgpu-waves-per-eu"="2"
% W8-SAME: "amdgpu-waves-per-eu"="8"
% NONE-NOT: "amdgpu-waves-per-eu"
% NCS-SAME: "no_callee_saved_registers"
% KEEPS-NOT: "no_callee_saved_registers"
% CHECK-SAME: }

% ERROR: error: -fgpu-process-waves=33: expected the waves a SIMD, from 0 to 32

% DRIVER: "-triple" "amdgpu10.30-amd-amdhsa"{{.*}} "-fprocess-kernels" "-fgpu-process-waves=6" "-fgpu-process-no-callee-saved"
% DRIVER-GFX90A: "-triple" "amdgpu9.0a-amd-amdhsa"{{.*}} "-fprocess-kernels" "-fgpu-process-waves=2" "-fgpu-process-no-callee-saved"
% DRIVER-GFX1030: "-triple" "amdgpu10.30-amd-amdhsa"{{.*}} "-fprocess-kernels"
% DRIVER-GFX1030-NOT: -fgpu-process
% DRIVER-OVERRIDE: "-triple" "amdgpu9.0a-amd-amdhsa"{{.*}} "-fprocess-kernels" "-fgpu-process-waves=0" "-fgpu-process-callee-saved"
% DRIVER-OVERRIDE-NOT: no-callee-saved

:- module process_place_kernel.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
main(!IO) :- io.write_string("hello\n", !IO).
