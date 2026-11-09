% A receive over a selector of several mailboxes in a target region is
% library/process.m's receive/5, which is not built in for the device
% (docs/INPROGRESS.md, The device's side of the first slice, Decision 4):
% in the device's compile, the error is at the selector's closures, which
% the device cannot make, before the call. On the host it runs.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-mlir %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_receive_selector.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, maybe, process.

:- func double(int) = int.
double(N) = 2 * N.

main(!IO) :-
    new_mailbox(A, !IO),
    new_mailbox(B, !IO),
    send(B, 4, !IO),
    %$omp target
    (
        receive(selecting(A, double, selecting(B, double, new_selector)),
            after(0), Got, !IO),
% CHECK: omp-target-receive-selector.m:[[@LINE-2]]:30: error: 'merk.closure' op makes a closure, which cannot be done on an OpenMP device yet
% CHECK: omp-target-receive-selector.m:[[@LINE-3]]:51: error: 'merk.closure' op makes a closure, which cannot be done on an OpenMP device yet
        (
            Got = message(N)
        ;
            Got = timeout,
            N = 0
        )
    ),
    io.print_line(N, !IO).

% OUTPUT:8
