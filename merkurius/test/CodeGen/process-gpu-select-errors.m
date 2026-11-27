% Receive goals that a process kind that runs on a GPU cannot make yet
% (docs/INPROGRESS.md, Concurrency in merk, step 1). The device's runtime
% keeps at most 8 cases of one receive (device.cpp, kMostSelectCases), one
% for each mailbox, as desugaring makes a mailbox's selects one case. A
% selected message that holds an array or a cell may be in what its case
% gives, which the merk.free pass does not follow, so only words and flat
% records are selected (Lowering.cpp, CheckSelects;
% Inputs/process_gpu_select_array.m). On the host, neither is an error.
%
% REQUIRES: amdgpu-registered-target
% RUN: not merkc -fc1 -freceive-goal -fprocess-declarations -Wno-unselected-message -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o /dev/null 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:
% RUN: not merkc -fc1 -freceive-goal -fprocess-declarations -Wno-unselected-message -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %S/Inputs/process_gpu_select_array.m -o /dev/null 2>&1 \
% RUN:   | FileCheck %s --check-prefix=ARRAY --implicit-check-not=error:
% RUN: merkc -fc1 -freceive-goal -fprocess-declarations -Wno-unselected-message -emit-llvm %s -o /dev/null
% RUN: merkc -fc1 -freceive-goal -fprocess-declarations -Wno-unselected-message -emit-llvm %S/Inputs/process_gpu_select_array.m -o /dev/null

:- module process_gpu_select_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, process.

:- process nine(mailbox(int), mailbox(int), mailbox(int), mailbox(int),
    mailbox(int), mailbox(int), mailbox(int), mailbox(int))
    receives int on gpu.

nine(B1, B2, B3, B4, B5, B6, B7, B8, Self, !IO) :-
    receive [io(!IO)] (
% CHECK: process-gpu-select-errors.m:[[@LINE-1]]:5: error: 'merk.receive' op selects with more than 8 cases in one receive goal, which cannot be done in a GPU process yet
        select(Self, N) -> send(B1, N, !IO)
    ;   select(B1, N) -> send(B1, N, !IO)
    ;   select(B2, N) -> send(B1, N, !IO)
    ;   select(B3, N) -> send(B1, N, !IO)
    ;   select(B4, N) -> send(B1, N, !IO)
    ;   select(B5, N) -> send(B1, N, !IO)
    ;   select(B6, N) -> send(B1, N, !IO)
    ;   select(B7, N) -> send(B1, N, !IO)
    ;   select(B8, N) -> send(B1, N, !IO)
    ).

main(!IO) :-
    new_mailbox(Report, !IO),
    spawn(gpu(0), nine(Report, Report, Report, Report, Report, Report,
        Report, Report), _, !IO).

% ARRAY: process_gpu_select_array.m:12:5: error: 'merk.select' op not yet implemented: selecting messages that hold arrays or cells in a GPU process
