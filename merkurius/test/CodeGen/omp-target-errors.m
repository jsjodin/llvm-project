% What target regions cannot do yet: map values that are not scalars, or
% take clauses; and the other directives.
%
% RUN: not merkc -fc1 -fopenmp -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:

:- module omp_target_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, string.

:- pred len(string::in, int::out) is det.
len(S, N) :-
    %$omp target
    N = string.length(S).
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for values of type string in target regions

:- pred next(int::in, int::out) is det.
next(X, Y) :-
    %$omp target map(tofrom: X)
    Y = X + 1.
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for clauses of the target directive

:- pred twice(int::in, int::out) is det.
twice(X, Y) :-
    %$omp parallel
    Y = X * 2.
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for the parallel directive

main(!IO) :-
    len("abc", N),
    next(N, M),
    twice(M, K),
    io.write_int(K, !IO).
