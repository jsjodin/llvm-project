% Type classes across modules: instances of another module, abstract or not, with
% superclasses two levels up, go through their tables, whose symbols are that
% module's. Also methods of two modes, one defined by name, outputs in memory, a
% semidet method with outputs, a method's own type variable and constraint, a do
% loop whose body calls a method, and a tail call through a table, of 3,000,000
% iterations, which must not grow the stack. The output is mmc's for the same
% program but for the do loop, written as recursion, and with fewer iterations,
% since mmc's call there is not a tail call.
%
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %S/Inputs/tcshapes.m %s -o %t0 && %t0 | FileCheck %s --match-full-lines
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %S/Inputs/tcshapes.m %s -o %t0.O2 && %t0.O2 | FileCheck %s --match-full-lines
% RUN: merkc -fdo-loops -I %S/Inputs %S/Inputs/tcshapes.m %s -o %t && %t | FileCheck %s --match-full-lines

:- module typeclass_modules.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, float, list, maybe, tcshapes.

% Modes: a method with two modes, a by-name definition, many outputs in memory,
% semidet with outputs, a method's own constraint and type variable, io.
:- typeclass rich(T) where [
    pred twin(T, T),
    mode twin(in, out) is det,
    mode twin(out, in) is det,
    pred four(T::in, T::out, int::out, float::out, T::out) is det,
    pred pick(T::in, int::in, T::out) is semidet,
    pred with_other(T::in, U::in, U::out) is det <= sized(U),
    func wrap(T, V) = maybe(V),
    pred say(T::in, io::di, io::uo) is det,
    func flt(T) = float
].
:- instance sized(float) where [ size(F) = truncate_to_int(F) ].
:- instance rich(int) where [
    pred(twin/2) is int_twin,
    four(X, X + 1, X + 2, float(X), X + 3),
    pick(X, N, X + N) :- N > 0,
    with_other(X, U0, U) :- ( if X + size(U0) > 100 then U = U0 else U = U0 ),
    wrap(X, V) = ( if X > 0 then yes(V) else no ),
    say(X, !IO) :- io.write_int(X, !IO),
    flt(X) = float(X) / 2.0
].
:- instance rich(float) where [
    twin(X, X),
    four(X, X + 1.0, truncate_to_int(X), X * 2.0, X + 3.0),
    pick(X, N, X * float(N)) :- N > 1,
    with_other(_, U, U),
    wrap(_, V) = yes(V),
    say(X, !IO) :- io.write_float(X, !IO),
    flt(X) = X
].
:- instance rich(list(T)) <= rich(T) where [
    twin(X, X),
    four(L, L, length_of(L), 0.5, L),
    pick(L, N, L) :- length_of(L) > N,
    with_other(_, U, U),
    wrap(L, V) = ( L = [] -> no ; yes(V) ),
    say(L, !IO) :- say_all(L, !IO),
    flt(L) = sum_flt(L)
].
:- pred int_twin(int, int).
:- mode int_twin(in, out) is det.
:- mode int_twin(out, in) is det.
int_twin(X, X).

:- func length_of(list(T)) = int.
length_of([]) = 0.
length_of([_ | T]) = 1 + length_of(T).

:- pred say_all(list(T)::in, io::di, io::uo) is det <= rich(T).
say_all([], !IO).
say_all([X | Xs], !IO) :- say(X, !IO), io.write_string(",", !IO), say_all(Xs, !IO).

:- func sum_flt(list(T)) = float <= rich(T).
sum_flt([]) = 0.0.
sum_flt([X | Xs]) = flt(X) + sum_flt(Xs).

:- pred try(T::in, io::di, io::uo) is det <= rich(T).
try(X, !IO) :-
    twin(X, A), twin(B, A),
    four(B, C, N, F, D),
    say(C, !IO), io.write_string(" ", !IO),
    io.write_int(N, !IO), io.write_string(" ", !IO),
    io.write_float(F, !IO), io.write_string(" ", !IO),
    say(D, !IO), io.write_string(" ", !IO),
    ( if pick(X, 2, P) then say(P, !IO) else io.write_string("none", !IO) ),
    io.write_string(" ", !IO),
    with_other(X, 7, Y), io.write_int(Y, !IO), io.write_string(" ", !IO),
    with_other(X, 1.5, Z), io.write_float(Z, !IO), io.write_string(" ", !IO),
    io.write(wrap(X, "s"), !IO), io.nl(!IO).

% A tail call through a method, a loop that must not grow the stack.
:- typeclass step(T) where [ pred go(T::in, int::in, int::out) is det ].
:- type looper ---> looper.
:- instance step(looper) where [
    go(L, N, R) :- ( if N = 0 then R = 0 else go(L, N - 1, R) )
].
:- pred tail(T::in, int::in, int::out) is det <= step(T).
tail(X, N, R) :- go(X, N, R).

% Superclasses two levels up, from another module.
:- pred up(T::in, int::out, int::out, int::out) is det <= deep(T).
up(X, A, B, C) :- A = depth(X), label(X, B), C = size(X).

% A loop in a constrained predicate, its body calling a method.
:- pred sum_loop(int::in, T::in, int::out) is det <= sized(T).
sum_loop(N, X, Sum) :-
    ( for(I, 1, N), fromto(0, S0, S1, Sum) do
        S1 = S0 + I * size(X)
    ).

% Comparing values whose type_info comes from the dictionary, through a list's.
:- pred same_list(list(T)::in, list(T)::in) is semidet <= sized(T).
same_list(X, Y) :- X = Y.

main(!IO) :-
    try(5, !IO),
    try(2.5, !IO),
    try([1, 2, 3], !IO),
    try([[0.5], []], !IO),
    tail(looper, 3000000, R), io.write_int(R, !IO), io.nl(!IO),
    up(token(4), A, B, C), io.write_int(A + 10 * B + 100 * C, !IO), io.nl(!IO),
    io.write_int(size([[1, 2], [3]]), !IO), io.nl(!IO),
    sum_loop(4, token(3), L), io.write_int(L, !IO), io.nl(!IO),
    ( if same_list([1], [1]) then io.write_string("eq\n", !IO) else io.write_string("ne\n", !IO) ),
    ( if same_list([1.5], [2.5]) then io.write_string("eq\n", !IO) else io.write_string("ne\n", !IO) ).

% CHECK: 6 7 5.0 8 7 7 1.5 yes("s")
% CHECK-NEXT: 3.5 2 5.0 5.5 5.0 7 1.5 yes("s")
% CHECK-NEXT: 1,2,3, 3 0.5 1,2,3, 1,2,3, 7 1.5 yes("s")
% CHECK-NEXT: 0.5,,, 2 0.5 0.5,,, none 7 1.5 yes("s")
% CHECK-NEXT: 0
% CHECK-NEXT: 483
% CHECK-NEXT: 6
% CHECK-NEXT: 30
% CHECK-NEXT: eq
% CHECK-NEXT: ne
