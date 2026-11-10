% Errors in the kernels' compile of process kinds that run on a GPU
% (docs/INPROGRESS.md, GPU processes without omp target, step 4b). In the
% first version a kind's kernel takes integers, floats, characters and
% mailboxes, and the kind is not polymorphic (decision D3). What the kinds
% reach is compiled for the GPU, where a closure cannot be made yet
% (Inputs/process_kernel_closure.m). A kind that runs on the host only is
% not compiled for the GPU, and is no error.
%
% REQUIRES: amdgpu-registered-target
% RUN: not merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o /dev/null 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:
% RUN: not merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %S/Inputs/process_kernel_closure.m -o /dev/null 2>&1 \
% RUN:   | FileCheck %s --check-prefix=CLOSURE --implicit-check-not=error:
% RUN: not merkc -fc1 -fprocess-declarations -fembed-process-image=%t.missing -emit-llvm %s -o /dev/null 2>&1 \
% RUN:   | FileCheck %s --check-prefix=OPTION
% RUN: not merkc -fc1 -fprocess-declarations -fembed-process-image=gfx1030:%t.missing -emit-llvm %s -o /dev/null 2>&1 \
% RUN:   | FileCheck %s --check-prefix=MISSING

:- module process_kernel_errors.
:- interface.
:- import_module io, list, process.
:- process lister(list(int)) receives int on gpu.
% CHECK: process-kernel-errors.m:[[@LINE-1]]:4: error: not yet implemented: code generation for a process kind that runs on a GPU and takes a value of type list.list(int): its kernel takes integers, floats, characters and mailboxes
:- process echo receives T on gpu.
% CHECK: process-kernel-errors.m:[[@LINE-1]]:4: error: not yet implemented: code generation for a process kind that runs on a GPU and is polymorphic
:- process host_lister(list(int)) receives int.
:- implementation.

lister(_, _, !IO).
echo(_, !IO).
host_lister(_, _, !IO).

% CLOSURE: process_kernel_closure.m:11:12: error: 'merk.closure' op makes a closure, which cannot be done in a GPU process yet
% CLOSURE: process_kernel_closure.m:14:9: error: 'merk.call_closure' op calls a closure, which cannot be done in a GPU process yet
% OPTION: error: -fembed-process-image={{.*}}.missing: expected the GPU and the image's file, as gfx1030:file
% MISSING: error: cannot read the GPU processes' image {{.*}}.missing: {{.*}}
