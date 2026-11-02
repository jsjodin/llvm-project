% Closures of predicates' and functions' names, as mmc makes them: a lambda
% expression whose body calls the predicate or function with the curried
% arguments and the lambda's. A closure passes the type_infos and
% typeclass_infos that call would, its arguments' types those of the
% curried arguments and then the closure type's, from the procedure's own,
% constants, cells, instances, superclasses, or what a goal gave, in a
% lambda expression and an instance method as elsewhere; a higher-order
% call passes none.
%
% RUN: merkc -fc1 -fdebug-dump-polymorphism %s 2>%t.err | FileCheck %s
% RUN: %if mmc %{ %compare-polymorphism %s %}

:- module closures.
:- interface.
:- import_module list.
:- typeclass sh(T) where [ pred sh(T::in, int::out) is det ].
:- typeclass sub(T) <= sh(T) where [ func sub(T) = int ].
:- instance sub(int).
:- typeclass two(A, B) where [ func two(A, B) = int ].
:- typeclass app(T) where [ pred app(T::in, int::out) is det ].
:- instance sh(int).
:- instance sh(list(T)) <= sh(T).
:- instance two(int, string).
:- instance app(list(T)) <= sh(T).
:- type box(T) ---> box(contents :: T).
:- some [T] pred cout(int::in, T::out) is det => sh(T).
:- pred use(T::in, int::out) is det <= sh(T).
:- pred id(T::in, T::out) is det.
:- pred two(T::in, U::in, int::out) is det.
:- func f(T) = list(T).
:- pred subuse(T::in, int::out) is det <= sub(T).
:- pred mono(int::in, int::out) is det.
:- pred closures(T::in, U::in, int::out) is det <= sub(U).
:- func f2(T, U) = list(U).
:- pred nested(T::in, int::out) is det <= sh(T).
:- pred given(int::out) is det.
:- pred methods(A::in, B::in, int::out) is det <= two(A, B).
:- pred fields(box(T)::in, T::out) is det.
:- pred apply(pred(A, int)::in(pred(in, out) is det), A::in, int::out) is det.
:- pred map2(pred(A, B)::in(pred(in, out) is det), list(A)::in, list(B)::out) is det.
:- implementation.
:- instance sh(int) where [ sh(X, X) ].
:- instance sub(int) where [ sub(X) = X ].
:- instance sh(list(T)) <= sh(T) where [ sh(_, 0) ].
:- instance two(int, string) where [ two(_, _) = 2 ].
:- instance app(list(T)) <= sh(T) where [
    (app(L, N) :- map2(use, L, Ns), P = use, P(L, _), N = list.length(Ns)) ].
cout(_, 3).
id(X, X).
two(_, _, 1).
f(X) = [X].
subuse(X, sub(X)).
mono(X, X).

% CHECK-LABEL: pred closures.closures/3 mode 0:
% CHECK-NEXT: hidden arguments: T, closures.sub(U)
% CHECK-NEXT: type_infos in typeclass_infos: U (type_info 2 of closures.sub(U))
% CHECK-NEXT: closure pred closures.id/2: T (argument)
% CHECK-NEXT: closure pred closures.two/3: T (argument), U (type_info 2 of closures.sub(U))
% CHECK-NEXT: closure pred closures.two/3: int (constant), string (constant)
% CHECK-NEXT: closure pred closures.use/2: closures.sh(U) (superclass 1 of closures.sub(U) (argument))
% CHECK-NEXT: closure pred closures.use/2: closures.sh(list.list(U)) (instance with closures.sh(U) (superclass 1 of closures.sub(U) (argument)))
% CHECK-NEXT: closure pred closures.use/2: closures.sh(int) (constant)
% CHECK-NEXT: closure pred closures.subuse/2: closures.sub(U) (argument)
% CHECK-NEXT: call pred closures.apply/3: U (type_info 2 of closures.sub(U))
% CHECK-NEXT: closure pred closures.sh/2: closures.sh(U) (superclass 1 of closures.sub(U) (argument))
% CHECK-NEXT: closure func closures.f/1: T (argument)
% CHECK-NEXT: closure pred closures.use/2: closures.sh(list.list(U)) (instance with closures.sh(U) (superclass 1 of closures.sub(U) (argument)))
% CHECK-NEXT: call pred closures.apply/3: list.list(U) (cell)
closures(T, U, N) :-
    P1 = id, P1(T, _),
    P2 = two(T), P2(U, _),
    P3 = two(1), P3("a", _),
    P4 = use, P4(U, _),
    P5 = use, P5([U], _),
    P6 = use, P6(3, _),
    P7 = subuse, apply(P7, U, _),
    P8 = sh, P8(U, _),
    P9 = mono, P9(1, _),
    F = f, _ = F(T),
    apply(use, [U], N).

use(X, N) :- sh(X, N).
f2(_, Y) = [Y].
apply(P, A, N) :- P(A, N).
map2(_, [], []).
map2(P, [X | Xs], [Y | Ys]) :- P(X, Y), map2(P, Xs, Ys).
% CHECK-LABEL: pred closures.nested/2 mode 0:
% CHECK-NEXT: hidden arguments: closures.sh(T)
% CHECK-NEXT: type_infos in typeclass_infos: T (type_info 1 of closures.sh(T))
% CHECK-NEXT: closure pred closures.use/2: closures.sh(list.list(T)) (instance with closures.sh(T) (argument))
nested(T, N) :-
    P = (pred(M::out) is det :- Q = use, Q([T], M)),
    P(N).
% CHECK-LABEL: pred closures.given/1 mode 0:
% CHECK-NEXT: call pred closures.cout/2: closures.sh(E[[C:[0-9]+]]) (output)
% CHECK-NEXT: closure pred closures.use/2: closures.sh(E[[C]]) (output of closures.cout)
% CHECK-NEXT: call pred closures.use/2: closures.sh(list.list(E[[C]])) (instance with closures.sh(E[[C]]) (output of closures.cout))
% CHECK-NEXT: closure pred closures.use/2: closures.sh(list.list(E[[C]])) (instance with closures.sh(E[[C]]) (output of closures.cout))
% CHECK-NEXT: call pred closures.apply/3: list.list(E[[C]]) (cell)
given(N) :-
    cout(1, C),
    P = use(C), P(_),
    Q = (pred(M::out) is det :- use([C], M)), Q(_),
    apply(use, [C], N).
% CHECK-LABEL: pred closures.methods/3 mode 0:
% CHECK-NEXT: hidden arguments: closures.two(A, B)
% CHECK-NEXT: type_infos in typeclass_infos: A (type_info 1 of closures.two(A, B)), B (type_info 2 of closures.two(A, B))
% CHECK-NEXT: closure func closures.two/2: closures.two(A, B) (argument)
methods(A, B, N) :-
    F = two, N = F(A, B).
% CHECK-LABEL: pred closures.fields/2 mode 0:
% CHECK-NEXT: hidden arguments: T
% CHECK-NEXT: application field closures.contents/1: T (argument)
% CHECK-NEXT: closure func closures.f2/2: closures.box(T) (cell), int (constant)
fields(B, X) :-
    X = B ^ contents,
    G = f2(B), _ = G(1).

% The instance's method, whose dump comes last.
% CHECK-LABEL: method closures.app(list.list(T)) pred closures.app/2 mode 0
% CHECK-NEXT: hidden arguments: closures.sh(T)
% CHECK-NEXT: type_infos in typeclass_infos: T (type_info 1 of closures.sh(T))
% CHECK-NEXT: closure pred closures.use/2: closures.sh(T) (argument)
% CHECK-NEXT: call pred closures.map2/3: T (type_info 1 of closures.sh(T)), int (constant)
% CHECK-NEXT: closure pred closures.use/2: closures.sh(list.list(T)) (instance with closures.sh(T) (argument))
