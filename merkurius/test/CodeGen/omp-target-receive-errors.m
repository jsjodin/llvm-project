% A target region receives only with any, and only one-word messages so
% far (docs/INPROGRESS.md, The device's side of the first slice, Decisions
% 3 and 4): another selector, or a message of another type, is an error at
% the receive, in the host's compile as in the device's. A selector over
% several mailboxes is omp-target-receive-selector.m.
%
% RUN: not merkc -fc1 -fopenmp -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-mlir %s 2>&1 | FileCheck %s --implicit-check-not=error:
% REQUIRES: amdgpu-registered-target

:- module omp_target_receive_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, maybe, process.

:- pred positive(int::in, int::out) is semidet.
positive(N, N) :-
    N > 0.

:- pred filtered(mailbox(int)::in, received(int)::out, io::di, io::uo) is det.
filtered(Ints, Got, !IO) :-
    %$omp target
    receive(Ints, positive, after(0), Got, !IO).
% CHECK: omp-target-receive-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for receiving in target regions with a selector other than any

:- pred greeting(mailbox(string)::in, io::di, io::uo) is det.
greeting(Strings, !IO) :-
    %$omp target
    receive(Strings, any, after(0), _, !IO).
% CHECK: omp-target-receive-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for receiving messages of type string in target regions

main(!IO) :-
    new_mailbox(Ints, !IO),
    filtered(Ints, _, !IO),
    new_mailbox(Strings, !IO),
    greeting(Strings, !IO).
