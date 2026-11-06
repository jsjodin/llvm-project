% What target regions cannot do yet: map values that are not scalars or
% arrays of them, make or fill arrays, which the device has no runtime
% for, or take clauses; and the other directives.
%
% RUN: not merkc -fc1 -fopenmp -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:

:- module omp_target_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int.

:- pred same(string::in, string::out) is det.
same(S, T) :-
    %$omp target
    T = S.
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for values of type string in target regions

:- pred first(array(string)::array_ui, string::out) is det.
first(A, S) :-
    %$omp target
    S = A ^ elem(0).
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for values of type array.array(string) in target regions

:- pred make(int::in, array(int)::array_uo) is det.
make(N, A) :-
    %$omp target
    A = array.init(N, 0).
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for making arrays in target regions

:- pred zero(array(int)::array_di, array(int)::array_uo) is det.
zero(!A) :-
    %$omp target
    array.fill(0, !A).
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for filling arrays in target regions

:- pred next(int::in, int::out) is det.
next(X, Y) :-
    %$omp target map(tofrom: X)
    Y = X + 1.
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for the map clause of the target directive

:- pred twice(int::in, int::out) is det.
twice(X, Y) :-
    %$omp parallel
    Y = X * 2.
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for the parallel directive

main(!IO) :-
    same("abc", _),
    N = 3,
    first(array.init(1, "a"), _),
    make(1, _),
    zero(array.init(1, 1), _),
    next(N, M),
    twice(M, K),
    io.write_int(K, !IO).
