% Foreign code that a GPU process cannot run is an error in the compile of
% its kernels, as an OpenMP device's is (omp-target-foreign-errors.m): a
% GPU process's image has no OpenMP device runtime, so a call of one of its
% functions is one, but omp_get_device_num, which the image's start object
% defines, is not.
%
% RUN: not merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=note:
% REQUIRES: amdgpu-registered-target

:- module process_foreign_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, process.

main(!IO) :-
    new_mailbox(Reply, !IO),
    spawn(gpu(0), worker(Reply), _, !IO),
    receive(Reply, any, after(60000), _, !IO).

:- process worker(mailbox(int)) receives int on gpu.

worker(Reply, _Self, !IO) :-
    send(Reply, thread(device(0)), !IO).

:- pragma foreign_decl("C", "
int omp_get_thread_num(void);
int omp_get_device_num(void);
").

:- func thread(int) = int.
:- pragma foreign_proc("C", thread(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
    "Y = X + omp_get_thread_num();").
% CHECK: process-foreign-errors.m:[[#@LINE-3]]:4: error: cannot call process_foreign_errors.thread_f1_0 in a GPU process yet: its foreign code calls omp_get_thread_num, which is not on the device

:- func device(int) = int.
:- pragma foreign_proc("C", device(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
    "Y = X + omp_get_device_num();").
