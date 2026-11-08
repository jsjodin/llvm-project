% What a closure's layout says of each thing it captured (design document,
% Closures that say what they captured).
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK

:- module closure_layout.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pred two(T::in, list(T)::in, int::in, int::out) is det.
two(_, L, A, B) :-
    B = A + list.length(L).

:- pred poly(T::in, list(T)::in, int::out) is det.
poly(X, L, R) :-
    P = two(X, L),
    % MERK-DAG: merk.closure @{{.*}}two{{.*}}$closure3{{.*}}captures ["type_info", array<i64: 0, 0>, ["list.list/1", array<i64: 0, 0>]]
    P(1, A),
    Q = (pred(Y::in, Z::out) is det :- Z = Y + list.length([X | L])),
    % MERK-DAG: merk.closure @{{.*}}$lambda{{.*}}captures ["type_info", array<i64: 0, 0>, ["list.list/1", array<i64: 0, 0>]]
    Q(A, R).

main(!IO) :-
    N = 2,
    P = (pred(Y::in, Z::out) is det :- Z = Y + N),
    % MERK-DAG: merk.closure @{{.*}}$lambda{{.*}}captures {{\[\["int/0"\]\]}}
    P(1, A),
    poly("x", ["y"], B),
    io.write_int(A + B, !IO),
    io.nl(!IO).
