% Foreign code that a device cannot run is an error in the device's compile
% (design document, Library code on the device), at the
% foreign_proc, naming what its C calls that the device does not have: a
% function that neither the GPU's libc nor its libm has, or the host's
% runtime, with a note for each call in the chain from the region. The
% host's compile is the ordinary one.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=note:
% RUN: merkc -fc1 -fopenmp -emit-llvm %s -o /dev/null
% REQUIRES: amdgpu-registered-target

:- module omp_target_foreign_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int.
:- pragma foreign_decl("C", "#include <unistd.h>").

main(!IO) :-
    %$omp target
    compute(N, X),
    io.write_int(N, !IO),
    io.nl(!IO),
    io.write_float(X, !IO),
    io.nl(!IO).

:- pred compute(int::out, float::out) is det.

compute(N, X) :-
    N = pid(1),
    X = checked(2.25).
% CHECK-DAG: omp-target-foreign-errors.m:[[#@LINE+4]]:4: error: cannot call omp_target_foreign_errors.checked_f1_0 on an OpenMP device yet: its foreign code calls merk_uncaught_exception, which is not on the device

    % The host's runtime, called by its own name, not the device's.
:- func checked(float) = float.
:- pragma foreign_proc("C", checked(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe], "
    extern void merk_uncaught_exception(const char *);
    if (X < 0.0) {
        merk_uncaught_exception(\"negative\");
    }
    Y = X;
").

:- func pid(int) = int.
:- pragma foreign_proc("C", pid(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe], "Y = X + getpid();").
% CHECK-DAG: omp-target-foreign-errors.m:[[#@LINE-2]]:4: error: cannot call omp_target_foreign_errors.pid_f1_0 on an OpenMP device yet: its foreign code calls getpid, which is not on the device
