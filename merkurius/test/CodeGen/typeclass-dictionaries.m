% Type classes: a typeclass_info for each constraint, instances' methods called
% through its table, directly when the instance is known, a superclass's taken
% out of one, a constrained instance's cell built from its premises, an
% instance's own type_info, and a type_info found in a typeclass_info. The output
% is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -O2 %s -o %t0.O2 && %t0.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: env MERK_HEAP_CHUNK_BYTES=64 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines

:- module typeclass_dictionaries.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, float, list.

% A class with a func, a pred and a semidet method, over several representations.
:- typeclass shape(T) where [
    func area(T) = float,
    pred name(T::in, string::out) is det,
    pred big(T::in) is semidet,
    pred scale(int::in, T::in, T::out) is det
].

:- type sq ---> sq(float).
:- type ci ---> ci(int, int).
:- type unit ---> unit.

:- instance shape(sq) where [
    area(sq(S)) = S * S,
    name(_, "square"),
    (big(sq(S)) :- S > 2.0),
    scale(K, sq(S), sq(float(K) * S))
].
:- instance shape(ci) where [
    (area(ci(R, _)) = 3.0 * float(R) * float(R)),
    name(ci(_, X), N) :- ( if X > 0 then N = "circle" else N = "disc" ),
    (big(_) :- fail),
    scale(K, ci(R, X), ci(K * R, X))
].
:- instance shape(int) where [
    area(X) = float(X),
    name(_, "int"),
    big(X) :- X > 10,
    scale(K, X, K * X)
].
:- instance shape(float) where [
    area(X) = X,
    name(_, "float"),
    big(X) :- X > 10.0,
    scale(K, X, float(K) * X)
].
:- instance shape(unit) where [
    area(_) = 0.5,
    name(_, "unit"),
    big(_) :- true,
    scale(_, U, U)
].
% A constrained instance, with a premise.
:- instance shape(list(T)) <= shape(T) where [
    area(L) = total(L),
    name(L, N) :- ( L = [H | _], name(H, N) ; L = [], N = "empty" ),
    big(L) :- all_big(L),
    scale(K, L0, L) :- scale_all(K, L0, L)
].
% An instance with a type variable no constraint has.
:- type box(A) ---> box(A, int).
:- instance shape(box(A)) where [
    area(box(_, N)) = float(N),
    name(box(_, _), "box"),
    big(box(_, N)) :- N > 1,
    scale(K, box(A, N), box(A, K * N))
].

:- func total(list(T)) = float <= shape(T).
total([]) = 0.0.
total([X | Xs]) = area(X) + total(Xs).

:- pred all_big(list(T)::in) is semidet <= shape(T).
all_big([]).
all_big([X | Xs]) :- big(X), all_big(Xs).

:- pred scale_all(int::in, list(T)::in, list(T)::out) is det <= shape(T).
scale_all(_, [], []).
scale_all(K, [X | Xs], [Y | Ys]) :- scale(K, X, Y), scale_all(K, Xs, Ys).

% A polymorphic predicate that uses a method and the type_info in the dictionary.
:- pred report(T::in, io::di, io::uo) is det <= shape(T).
report(X, !IO) :-
    name(X, N),
    io.write_string(N, !IO),
    io.write_string(" ", !IO),
    io.write_float(area(X), !IO),
    ( if big(X) then io.write_string(" big ", !IO) else io.write_string(" small ", !IO) ),
    scale(2, X, Y),
    io.write_float(area(Y), !IO),
    io.write_string(" ", !IO),
    io.write(Y, !IO),
    ( if X = Y then io.write_string(" same", !IO) else io.write_string(" differ", !IO) ),
    io.nl(!IO).

% A superclass, whose method a subclass's constraint reaches.
:- typeclass named(T) <= shape(T) where [
    func tag(T) = string
].
:- instance named(int) where [ tag(_) = "I" ].
:- instance named(list(T)) <= shape(T) where [ tag(_) = "L" ].

:- pred super_call(T::in, io::di, io::uo) is det <= named(T).
super_call(X, !IO) :-
    io.write_string(tag(X), !IO),
    io.write_string(":", !IO),
    report(X, !IO).

% Two parameters.
:- typeclass conv(A, B) where [ func conv(A) = B ].
:- instance conv(int, float) where [ conv(X) = float(X) ].
:- instance conv(float, int) where [ conv(X) = truncate_to_int(X) ].
:- instance conv(int, string) where [ conv(X) = ( if X > 0 then "pos" else "neg" ) ].

:- pred conv2(A::in, B::out) is det <= conv(A, B).
conv2(A, conv(A)).

main(!IO) :-
    report(sq(3.0), !IO),
    report(ci(2, 1), !IO),
    report(ci(1, 0), !IO),
    report(7, !IO),
    report(1.5, !IO),
    report(unit, !IO),
    report([1, 20], !IO),
    report([sq(3.0), sq(4.0)], !IO),
    report([[sq(3.0)], []], !IO),
    report([] : list(int), !IO),
    report(box("x", 2), !IO),
    report([box(1.5, 0)], !IO),
    super_call(11, !IO),
    super_call([1, 2, 3], !IO),
    conv2(3, F : float), io.write_float(F, !IO), io.nl(!IO),
    conv2(3.9, I : int), io.write_int(I, !IO), io.nl(!IO),
    conv2(42, S : string), io.write_string(S, !IO), io.nl(!IO).

% Tables come first; main makes the dictionaries polymorphic procedures need,
% which call through them.
% MERK: merk.instance @"merk.instance.typeclass_dictionaries.shape(int)" visibility "private" methods [@"typeclass_dictionaries.area_f1_0$typeclass_dictionaries.shape(int)$dispatch",
% MERK: merk.typeclass_info @"merk.instance.typeclass_dictionaries.shape(typeclass_dictionaries.sq)"
% MERK: merk.proc private @typeclass_dictionaries.super_call_3_0(%arg0: !merk.typeclass_info,
% MERK: merk.typeclass_info_word %arg0, 1 : !merk.typeclass_info
% MERK: merk.proc private @typeclass_dictionaries.total_f1_0(%arg0: !merk.typeclass_info,
% MERK: merk.call_method %arg0[0] (

% OUTPUT: square 9.0 big 36.0 sq(6.0) differ
% OUTPUT-NEXT: circle 12.0 small 48.0 ci(4, 1) differ
% OUTPUT-NEXT: disc 3.0 small 12.0 ci(2, 0) differ
% OUTPUT-NEXT: int 7.0 small 14.0 14 differ
% OUTPUT-NEXT: float 1.5 small 3.0 3.0 differ
% OUTPUT-NEXT: unit 0.5 big 0.5 unit same
% OUTPUT-NEXT: int 21.0 small 42.0 [2, 40] differ
% OUTPUT-NEXT: square 25.0 big 100.0 [sq(6.0), sq(8.0)] differ
% OUTPUT-NEXT: square 9.0 big 36.0 {{\[\[}}sq(6.0)], []] differ
% OUTPUT-NEXT: empty 0.0 big 0.0 [] same
% OUTPUT-NEXT: box 2.0 big 4.0 box("x", 4) differ
% OUTPUT-NEXT: box 0.0 small 0.0 [box(1.5, 0)] same
% OUTPUT-NEXT: I:int 11.0 big 22.0 22 differ
% OUTPUT-NEXT: L:int 6.0 small 12.0 [2, 4, 6] differ
% OUTPUT-NEXT: 3.0
% OUTPUT-NEXT: 3
% OUTPUT-NEXT: pos
