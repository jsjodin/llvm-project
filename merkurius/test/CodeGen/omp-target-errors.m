% What target regions cannot do yet: map values that are not scalars,
% arrays of them or terms whose cells hold only scalars, give terms, make
% or fill arrays, which the device has no runtime for, or take clauses;
% and the other directives.
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

:- type shape
    --->    circle(float, float)
    ;       square(float, float).

:- pred width(shape::in, float::out) is det.
width(Shape, W) :-
    %$omp target
    (
        Shape = circle(W, _)
    ;
        Shape = square(W, _)
    ).
% CHECK: omp-target-errors.m:[[@LINE-6]]:5: error: not yet implemented: code generation for values of type omp_target_errors.shape in target regions

:- type named
    --->    named(string, int).

:- pred number(named::in, int::out) is det.
number(Named, N) :-
    %$omp target
    Named = named(_, N).
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for values of type omp_target_errors.named in target regions

:- type pair
    --->    pair(int, int).

:- pred double(int::in, pair::out) is det.
double(X, P) :-
    %$omp target
    P = pair(X, X).
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for values of type omp_target_errors.pair in target regions

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
    width(circle(1.0, 2.0), _),
    number(named("a", 1), _),
    double(1, _),
    next(N, M),
    twice(M, K),
    io.write_int(K, !IO).
