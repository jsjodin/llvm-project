% Type classes' corners: a functional dependency, a method with no arguments,
% constraints on types built from type variables, a list's and a tuple's, whose
% variables' type_infos are their arguments, an inferred constraint, a class with
% no methods, a superclass with its parameters in another order, a method
% defined by a foreign_proc, and a semidet method with outputs in memory. The
% output is mmc's for each call but in_list's and in_tuple's, which kill mmc's
% programs at io.write, as it passes the type's type_info as the variable's (the
% design document's register).
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --match-full-lines
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines

:- module typeclass_corners.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, float, list.

% A functional dependency: E is only in the constraint of callers.
:- typeclass coll(C, E) <= (C -> E) where [
    func first(C) = E,
    func put(E, C) = C
].
:- type ints ---> ints(list(int)).
:- instance coll(ints, int) where [
    first(ints(L)) = ( L = [H | _] -> H ; 0 ),
    put(X, ints(L)) = ints([X | L])
].
:- type fs(T) ---> fs(list(T)).
:- instance coll(fs(T), list(T)) <= zero(T) where [
    first(fs(L)) = ( L = [H | _] -> [H, zero] ; [zero] ),
    put(X, fs(L)) = fs(app(X, L))
].
:- func app(list(T), list(T)) = list(T).
app([], Y) = Y.
app([H | T], Y) = [H | app(T, Y)].
:- typeclass zero(T) where [ func zero = T ].
:- instance zero(int) where [ zero = 0 ].
:- instance zero(float) where [ zero = 0.0 ].
:- instance zero(list(T)) where [ zero = [] ].

:- pred grow(C::in, E::in, C::out, E::out) is det <= coll(C, E).
grow(C0, E, C, F) :- C = put(E, C0), F = first(C).

% A method with no arguments, whose instance only the result chooses.
:- func zeros = {int, float}.
zeros = {zero, zero}.

% A constraint on a type built from a variable: T's type_info is an argument of
% the type's, in a list's cell and in a tuple's.
:- typeclass sh(T) where [ func sz(T) = int ].
:- instance sh(list(T)) where [ sz(L) = len(L) ].
:- instance sh({A, B}) where [ sz(_) = 2 ].
:- func len(list(T)) = int.
len([]) = 0.
len([_ | T]) = 1 + len(T).

:- pred in_list(list(T)::in, T::in, io::di, io::uo) is det <= sh(list(T)).
in_list(L, X, !IO) :-
    io.write_int(sz(L), !IO), io.write_string(" ", !IO),
    io.write(X, !IO), io.write_string(" ", !IO),
    ( if [X] = L then io.write_string("one", !IO) else io.write_string("many", !IO) ),
    io.nl(!IO).

:- pred in_tuple(A::in, B::in, io::di, io::uo) is det <= sh({A, B}).
in_tuple(A, B, !IO) :-
    io.write_int(sz({A, B}), !IO), io.write_string(" ", !IO),
    io.write(A, !IO), io.write_string(" ", !IO),
    io.write(B, !IO), io.write_string(" ", !IO),
    ( if A = A, B = B then io.write_string("eq", !IO) else true ),
    io.nl(!IO).

% Inferred: no declaration, and its constraint comes from its body.
inferred(X, Y) = sz(X) + sz(Y).

% No methods: a constraint that carries only a type_info.
:- typeclass mark(T) where [].
:- instance mark(int) where [].
:- instance mark(float) where [].
:- pred marked(T::in, T::in, io::di, io::uo) is det <= mark(T).
marked(X, Y, !IO) :-
    ( if X = Y then io.write_string("same", !IO) else io.write_string("diff", !IO) ),
    io.write_string(" ", !IO), io.write(X, !IO), io.nl(!IO).

% A superclass with a different order of parameters, and a method defined by a
% predicate of a foreign_proc.
:- typeclass base(A, B) where [ func comb(A, B) = int ].
:- typeclass sub(X, Y) <= base(Y, X) where [ func other(X, Y) = int ].
:- instance base(int, float) where [ comb(A, B) = A + truncate_to_int(B) ].
:- instance sub(float, int) where [ func(other/2) is c_other ].
:- func c_other(float, int) = int.
:- pragma foreign_proc("C", c_other(A::in, B::in) = (R::out),
    [will_not_call_mercury, promise_pure], "R = (MR_Integer) A * 10 + B;").
:- func use(X, Y) = int <= sub(X, Y).
use(X, Y) = other(X, Y) + comb(Y, X).

% A semidet method with outputs in memory.
:- typeclass parts(T) where [
    pred split(T::in, int::out, int::out, int::out) is semidet
].
:- instance parts(int) where [
    split(X, X, X + 1, X + 2) :- X > 0
].
:- instance parts(list(T)) <= parts(T) where [
    (split([H | _], A, B, C) :- split(H, A, B, C))
].
:- func try_split(T) = int <= parts(T).
try_split(X) = R :- ( if split(X, A, B, C) then R = A + 10 * B + 100 * C else R = -1 ).

main(!IO) :-
    grow(ints([1]), 5, C1, F1), io.write(C1, !IO), io.write_int(F1, !IO), io.nl(!IO),
    grow(fs([2.5]), [3.5], C2, F2), io.write(C2, !IO), io.write(F2, !IO), io.nl(!IO),
    grow(fs([[1]]), [[]], C3, F3), io.write(C3, !IO), io.write(F3, !IO), io.nl(!IO),
    io.write(zeros, !IO), io.nl(!IO),
    in_list([1, 2], 3, !IO),
    in_list(["a"], "a", !IO),
    in_tuple(1.5, "x", !IO),
    in_tuple('c', {1, 2}, !IO),
    io.write_int(inferred([1], {2, 3}), !IO), io.nl(!IO),
    marked(1, 1, !IO),
    marked(1.5, 2.5, !IO),
    io.write_int(use(2.5, 4), !IO), io.nl(!IO),
    io.write_int(try_split(3), !IO), io.nl(!IO),
    io.write_int(try_split([0]), !IO), io.nl(!IO),
    io.write_int(try_split([[7]]), !IO), io.nl(!IO),
    io.write_int(sz([1, 2, 3]), !IO), io.nl(!IO).
% A known instance's method is called directly, with the instance's type_info.
% MERK: merk.call @"typeclass_corners.sz_f1_0$typeclass_corners.sh(list.list(T))"(%{{[0-9]+}}, %{{[0-9]+}})

% CHECK: ints([5, 1])5
% CHECK-NEXT: fs([3.5, 2.5])[3.5, 0.0]
% CHECK-NEXT: fs({{\[\[}}], [1]]){{\[\[}}], []]
% CHECK-NEXT: {0, 0.0}
% CHECK-NEXT: 2 3 many
% CHECK-NEXT: 1 "a" one
% CHECK-NEXT: 2 1.5 "x" eq
% CHECK-NEXT: 2 'c' {1, 2} eq
% CHECK-NEXT: 3
% CHECK-NEXT: same 1
% CHECK-NEXT: diff 1.5
% CHECK-NEXT: 30
% CHECK-NEXT: 543
% CHECK-NEXT: -1
% CHECK-NEXT: 987
% CHECK-NEXT: 3
