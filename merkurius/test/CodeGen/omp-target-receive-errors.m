% A target region receives only with any so far (docs/INPROGRESS.md, The
% device's side of the first slice, Decision 4), and messages of any ground
% type whose arguments are not higher-order (GPU processes without omp
% target, step 4e): another selector, or a message of a type with a closure
% in its arguments, is an error at the receive, in the host's compile as in
% the device's. A selector over
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
:- import_module int, list, maybe, process.

:- pred positive(int::in, int::out) is semidet.
positive(N, N) :-
    N > 0.

:- pred filtered(mailbox(int)::in, received(int)::out, io::di, io::uo) is det.
filtered(Ints, Got, !IO) :-
    %$omp target
    receive(Ints, positive, after(0), Got, !IO).
% CHECK: omp-target-receive-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for receiving in target regions with a selector other than any

:- pred tests(mailbox(list(pred(int)))::in, io::di, io::uo) is det.
tests(Tests, !IO) :-
    %$omp target
    receive(Tests, any, after(0), _, !IO).
% CHECK: omp-target-receive-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for receiving messages of type list.list(pred(int)) in target regions

main(!IO) :-
    new_mailbox(Ints, !IO),
    filtered(Ints, _, !IO),
    new_mailbox(Tests, !IO),
    tests(Tests, !IO).
