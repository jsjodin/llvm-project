% Errors in the kernels' compile of process kinds that run on a GPU
% (docs/INPROGRESS.md, GPU processes without omp target, steps 4b and 4c).
% In the first version a kind's kernel takes integers, floats, characters
% and mailboxes, and the kind is not polymorphic (decision D3). An argument
% of another type has no instance of process.gpu_argument, through which
% spawn_Kind_at gives it to the kernel (Inputs/process_kernel_list.m). What
% the kinds reach is compiled for the GPU, where a closure cannot be made
% (Inputs/process_kernel_closure.m), which the check of GPU code reports
% (test/Semantics/processes/gpu-flat.m). A kind that runs on the host only
% is not compiled for the GPU, and is no error.
%
% REQUIRES: amdgpu-registered-target
% RUN: not merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o /dev/null 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:
% RUN: not merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %S/Inputs/process_kernel_closure.m -o /dev/null 2>&1 \
% RUN:   | FileCheck %s --check-prefix=CLOSURE --implicit-check-not=error:
% RUN: not merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %S/Inputs/process_kernel_list.m -o /dev/null 2>&1 \
% RUN:   | FileCheck %s --check-prefix=LIST --implicit-check-not=error:
% RUN: not merkc -fc1 -fprocess-declarations -fembed-process-image=%t.missing -emit-llvm %s -o /dev/null 2>&1 \
% RUN:   | FileCheck %s --check-prefix=OPTION
% RUN: not merkc -fc1 -fprocess-declarations -fembed-process-image=gfx1030:%t.missing -emit-llvm %s -o /dev/null 2>&1 \
% RUN:   | FileCheck %s --check-prefix=MISSING

:- module process_kernel_errors.
:- interface.
:- import_module io, list, process.
:- process echo receives T on gpu.
% CHECK: process-kernel-errors.m:[[@LINE-1]]:4: error: not yet implemented: code generation for a process kind that runs on a GPU and is polymorphic
:- process host_lister(list(int)) receives int.
:- implementation.

echo(_, !IO).
host_lister(_, _, !IO).

% CLOSURE: process_kernel_closure.m:11:12: error: code of a process kind that runs on a GPU must make only flat values, but this makes Add of type pred(int, int): pred(int, int) is a higher-order type
% LIST: process_kernel_list.m:6:4: error: unsatisfiable type class constraint `process.gpu_argument(list.list(int))'
% LIST: process_kernel_list.m:6:4: because: a process kind that runs on a GPU takes integers, floats, characters and mailboxes so far
% OPTION: error: -fembed-process-image={{.*}}.missing: expected the GPU and the image's file, as gfx1030:file
% MISSING: error: cannot read the GPU processes' image {{.*}}.missing: {{.*}}
