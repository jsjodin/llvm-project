% Existentially typed constructors: a cell has, before its arguments, a
% type_info for each existentially quantified type variable no constraint has,
% and then a typeclass_info for each constraint, as mmc's cells have them. A
% construction stores what polymorphism analysis passes, and a deconstruction
% gives them to the goals after it. A unification fails if the types differ.
% A comparison compares an argument's existential types, in mmc's order of
% types, just before its value, so an argument before it decides first.
% io.write finds an argument's type in the cell. The output is mmc's for the
% same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=UNIFY
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: env MERK_HEAP_CHUNK_BYTES=64 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines

:- module existential_constructors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- typeclass sh(T) where [ func sz(T) = int, func nm(T) = string ].
:- typeclass big(T) <= sh(T) where [ func bg(T) = int ].
:- instance sh(int) where [ sz(X) = X, nm(_) = "int" ].
:- instance sh(string) where [ sz(_) = 2, nm(_) = "string" ].
:- instance sh(list(T)) <= sh(T) where [ sz(L) = len(L), nm(_) = "list" ].
:- instance big(int) where [ bg(X) = X * 100 ].
:- func len(list(T)) = int <= sh(T).
len([]) = 0.
len([H | T]) = sz(H) + len(T).

% A typeclass_info, then the argument.
:- type box ---> some [T] box(T) => sh(T).
% A type_info, among a constant's constructors.
:- type any ---> none ; some [T] any(T).
% A constraint with a superclass, whose type_info is after its typeclass_info's.
:- type bigbox ---> some [T] bigbox(T, int) => big(T).
% Two variables, one constrained, and a variable inside an argument's type.
:- type pair2 ---> some [A, B] pair2(A, list(B)) => sh(A).
% A type parameter beside an existential variable.
:- type tagged(K) ---> some [V] tagged(K, V).
% An existentially typed constructor's term inside another's, at its variable.
:- type nest ---> some [B] nest(tagged(B)).

:- pred show(box::in, io::di, io::uo) is det.
show(box(X), !IO) :-
    io.write_string(nm(X), !IO), io.write_string("/", !IO),
    io.write_int(sz(X), !IO), io.nl(!IO).

:- func describe(any) = string.
describe(none) = "none".
describe(any(_)) = "any".

:- func bigsum(bigbox) = int.
bigsum(bigbox(X, N)) = bg(X) + sz(X) + N.

:- func pairsize(pair2) = int.
pairsize(pair2(A, Bs)) = sz(A) + count(Bs).

:- func count(list(T)) = int.
count([]) = 0.
count([_ | T]) = 1 + count(T).

:- pred cmps(T::in, list(T)::in, io::di, io::uo) is det.
cmps(_, [], !IO).
cmps(X, [Y | Ys], !IO) :- cmp(X, Y, !IO), cmps(X, Ys, !IO).

:- pred cmp(T::in, T::in, io::di, io::uo) is det.
cmp(X, Y, !IO) :-
    compare(R, X, Y),
    io.write(X, !IO), io.write_string(" ", !IO), io.write(R, !IO),
    io.write_string(" ", !IO), io.write(Y, !IO),
    ( if X = Y then io.write_string(" equal", !IO) else true ),
    io.nl(!IO).

main(!IO) :-
    B1 = 'new box'(42),
    B2 = 'new box'("hi"),
    B3 = 'new box'([1, 2, 3]),
    show(B1, !IO),
    show(B2, !IO),
    show(B3, !IO),
    io.write(B1, !IO), io.nl(!IO),
    io.write(B3, !IO), io.nl(!IO),
    A1 = 'new any'(3.5),
    A2 = 'new any'("a"),
    A3 = 'new any'(1),
    A4 = 'new any'([1]),
    io.write_string(describe(A1), !IO), io.nl(!IO),
    io.write_string(describe(none), !IO), io.nl(!IO),
    io.write([A1, A2, A3, none, A4], !IO), io.nl(!IO),
    ( if A1 = any(V) then io.write(V, !IO), io.nl(!IO) else true ),
    io.write_int(bigsum('new bigbox'(7, 1)), !IO), io.nl(!IO),
    P = 'new pair2'("abc", [1.5, 2.5]),
    io.write_int(pairsize(P), !IO), io.nl(!IO),
    io.write(P, !IO), io.nl(!IO),
    T1 = 'new tagged'("k", 1),
    T2 = 'new tagged'("k", 'c'),
    io.write(T1, !IO), io.nl(!IO),
    cmp(B1, B2, !IO),
    cmp(B1, 'new box'(42), !IO),
    cmp(B1, 'new box'(43), !IO),
    cmp(B2, B3, !IO),
    L = [A1, A2, A3, none, A4, 'new any'(2), 'new any'('x')],
    cmps(A3, L, !IO),
    cmps(none, L, !IO),
    cmp(T1, T2, !IO),
    cmp(T1, 'new tagged'("j", 1), !IO),
    cmp(T1, 'new tagged'("k", 1), !IO),
    % An argument before an existentially typed one decides first.
    cmp('new tagged'("j", 1), T2, !IO),
    cmp(P, 'new pair2'("abc", [1, 2]), !IO),
    cmp(P, 'new pair2'(1, [1.5, 2.5]), !IO),
    cmp(P, 'new pair2'("abc", [1.5, 2.5]), !IO),
    cmp('new pair2'("abc", [1]), 'new pair2'("abd", [1.5]), !IO),
    N1 = 'new nest'('new tagged'(1, "x")),
    N2 = 'new nest'('new tagged'(1, 2)),
    N3 = 'new nest'('new tagged'("a", 2)),
    io.write(N1, !IO), io.nl(!IO),
    cmp(N1, N2, !IO),
    cmp(N2, N3, !IO),
    cmp(N1, 'new nest'('new tagged'(1, "x")), !IO).

% OUTPUT: int/42
% OUTPUT-NEXT: string/2
% OUTPUT-NEXT: list/6
% OUTPUT-NEXT: box(42)
% OUTPUT-NEXT: box([1, 2, 3])
% OUTPUT-NEXT: any
% OUTPUT-NEXT: none
% OUTPUT-NEXT: [any(3.5), any("a"), any(1), none, any([1])]
% OUTPUT-NEXT: 3.5
% OUTPUT-NEXT: 708
% OUTPUT-NEXT: 4
% OUTPUT-NEXT: pair2("abc", [1.5, 2.5])
% OUTPUT-NEXT: tagged("k", 1)
% OUTPUT-NEXT: box(42) '<' box("hi")
% OUTPUT-NEXT: box(42) '=' box(42) equal
% OUTPUT-NEXT: box(42) '<' box(43)
% OUTPUT-NEXT: box("hi") '<' box([1, 2, 3])
% OUTPUT-NEXT: any(1) '>' any(3.5)
% OUTPUT-NEXT: any(1) '<' any("a")
% OUTPUT-NEXT: any(1) '=' any(1) equal
% OUTPUT-NEXT: any(1) '>' none
% OUTPUT-NEXT: any(1) '<' any([1])
% OUTPUT-NEXT: any(1) '<' any(2)
% OUTPUT-NEXT: any(1) '>' any('x')
% OUTPUT-NEXT: none '<' any(3.5)
% OUTPUT-NEXT: none '<' any("a")
% OUTPUT-NEXT: none '<' any(1)
% OUTPUT-NEXT: none '=' none equal
% OUTPUT-NEXT: none '<' any([1])
% OUTPUT-NEXT: none '<' any(2)
% OUTPUT-NEXT: none '<' any('x')
% OUTPUT-NEXT: tagged("k", 1) '>' tagged("k", 'c')
% OUTPUT-NEXT: tagged("k", 1) '>' tagged("j", 1)
% OUTPUT-NEXT: tagged("k", 1) '=' tagged("k", 1) equal
% OUTPUT-NEXT: tagged("j", 1) '<' tagged("k", 'c')
% OUTPUT-NEXT: pair2("abc", [1.5, 2.5]) '<' pair2("abc", [1, 2])
% OUTPUT-NEXT: pair2("abc", [1.5, 2.5]) '>' pair2(1, [1.5, 2.5])
% OUTPUT-NEXT: pair2("abc", [1.5, 2.5]) '=' pair2("abc", [1.5, 2.5]) equal
% OUTPUT-NEXT: pair2("abc", [1]) '<' pair2("abd", [1.5])
% OUTPUT-NEXT: nest(tagged(1, "x"))
% OUTPUT-NEXT: nest(tagged(1, "x")) '>' nest(tagged(1, 2))
% OUTPUT-NEXT: nest(tagged(1, 2)) '<' nest(tagged("a", 2))
% OUTPUT-NEXT: nest(tagged(1, "x")) '=' nest(tagged(1, "x")) equal

% MERK-DAG: merk.type_def "existential_constructors.box" [#merk.constructor<"box"(!merk.typeclass_info, !merk.word) hidden 1>]
% MERK-DAG: merk.type_def "existential_constructors.any" [#merk.constructor<"none">, #merk.constructor<"any"(!merk.type_info, !merk.word) hidden 1>]
% MERK-DAG: merk.type_def "existential_constructors.bigbox" [#merk.constructor<"bigbox"(!merk.typeclass_info, !merk.word, i64) hidden 1>]
% MERK-DAG: merk.type_def "existential_constructors.pair2" [#merk.constructor<"pair2"(!merk.type_info, !merk.typeclass_info, !merk.word, !merk.term<"list.list(T536870913)">) hidden 2>]
% MERK-DAG: merk.type_def "existential_constructors.tagged(T0)" [#merk.constructor<"tagged"(!merk.type_info, !merk.word, !merk.word) hidden 1>]
% MERK-DAG: merk.construct #merk.functor<"box", 1>(%{{.*}}, %{{.*}} : !merk.typeclass_info, !merk.word) -> !merk.term<"existential_constructors.box">
% A field's type is where the cell has its type_info: the field, and the word
% of it in a typeclass_info, after its method table.
% MERK-DAG: merk.type_ctor "existential_constructors.box/0" {{.*}} functors {{\[\[}}"box", array<i64: 0, 1>]]
% MERK-DAG: merk.type_ctor "existential_constructors.pair2/0" {{.*}} functors {{\[\[}}"pair2", array<i64: 1, 1>, ["list.list/1", array<i64: 0, 0>]]]
% MERK-DAG: merk.type_ctor "existential_constructors.tagged/1" {{.*}} functors {{\[\[}}"tagged", 0, array<i64: 0, 0>]]
% The types first, then the values.
% UNIFY-LABEL: merk.proc private @unify.existential_constructors.any(
% UNIFY: %[[X:[0-9]+]]:2 = merk.deconstruct #merk.functor<"any", 1> %arg0
% UNIFY: %[[Y:[0-9]+]]:2 = merk.deconstruct #merk.functor<"any", 1> %arg1
% UNIFY: %[[C:[0-9]+]] = merk.compare_type_info %[[X]]#0, %[[Y]]#0
% UNIFY: %[[E:[0-9]+]] = arith.cmpi eq, %[[C]]
% UNIFY: merk.guard %[[E]]
% UNIFY: merk.unify_word %[[X]]#0, %[[X]]#1, %[[Y]]#1
