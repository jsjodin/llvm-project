% Where a procedure's type_infos come from, as in mmc: one hidden argument
% for each type variable of its signature, in the order of their first
% appearance in its declaration, used or not, and expanded through
% equivalence types; at a call, for each of the callee's type variables, the
% caller's own argument for one of its type variables, a constant for a
% type without them, void among them, and a cell built from the type's
% arguments' type_infos for another. A function application, a field
% access and a call in a lambda expression pass them too, and a test of
% two values of a type with type variables needs its type's. A predicate
% whose signature is inferred has hidden arguments in the order of mmc's
% inferred signature. The
% procedures that make a closure of a polymorphic predicate are left to a
% later step.
%
% RUN: merkc -fc1 -fdebug-dump-polymorphism -finfer-modes %s 2>%t.err | FileCheck %s
% RUN: %if mmc %{ %compare-polymorphism --infer %s %}

:- module type_infos.
:- interface.
:- import_module list, pair.
:- type box(T) ---> box(contents :: T, size :: int).
:- type e(T) == pair(T, T).
:- pred two(T::in, U::in, pair(U, T)::out) is det.
% CHECK-LABEL: pred type_infos.two/3 mode 0:
% CHECK-NEXT: type_info arguments: T, U
:- pred rev(U::in, T::in, pair(U, T)::out) is det.
% CHECK-LABEL: pred type_infos.rev/3 mode 0:
% CHECK-NEXT: type_info arguments: U, T
:- func f(T, U) = pair(T, U).
% CHECK-LABEL: func type_infos.f/2 mode 0:
% CHECK-NEXT: type_info arguments: T, U
:- pred eqv(e(T)::in, e(T)::out) is det.
% CHECK-LABEL: pred type_infos.eqv/2 mode 0:
% CHECK-NEXT: type_info arguments: T
:- pred calls(int::out) is det.
% CHECK-LABEL: pred type_infos.calls/1 mode 0:
% CHECK-NEXT: call pred type_infos.two/3: int (constant), string (constant)
% CHECK-NEXT: call pred type_infos.ret/1: list.list(int) (constant)
% CHECK-NEXT: call pred type_infos.unused/2: pair.pair(string, int) (constant)
% CHECK-NEXT: call pred type_infos.unused/2: {int, character} (constant)
% CHECK-NEXT: application func type_infos.f/2: int (constant), list.list(int) (constant)
% CHECK-NEXT: call pred type_infos.unused/2: pair.pair(int, list.list(int)) (constant)
% CHECK-NEXT: call pred type_infos.eqv/2: int (constant)
% CHECK-NEXT: application func list.length/1: list.list(int) (constant)
:- pred poly(T::in, list(pair(T, int))::out) is det.
% CHECK-LABEL: pred type_infos.poly/2 mode 0:
% CHECK-NEXT: type_info arguments: T
% CHECK-NEXT: call pred type_infos.two/3: T (argument), list.list(T) (cell)
% CHECK-NEXT: call pred type_infos.unused/2: pair.pair(T, T) (cell)
% CHECK-NEXT: call pred builtin.compare/3: list.list(T) (cell)
:- pred undetermined(int::out) is det.
% CHECK-LABEL: pred type_infos.undetermined/1 mode 0:
% CHECK-NEXT: call pred type_infos.ret/1: void (constant)
% CHECK-NEXT: call pred type_infos.unused/2: list.list(void) (constant)
:- pred tests(T::in, list(T)::in, int::out) is det.
% CHECK-LABEL: pred type_infos.tests/3 mode 0:
% CHECK-NEXT: type_info arguments: T
% CHECK-NEXT: unify X = Y: T (argument)
% CHECK-NEXT: unify L = M: list.list(T) (cell)
% CHECK-NEXT: unify Y = X: T (argument)
% CHECK-NEXT: unify M = L: list.list(T) (cell)
:- pred fields(box(T)::in, T::in, box(T)::out) is det.
% CHECK-LABEL: pred type_infos.fields/3 mode 0:
% CHECK-NEXT: type_info arguments: T
% CHECK-NEXT: application field type_infos.contents/1: T (argument)
% CHECK-NEXT: application field update type_infos.'contents :='/2: T (argument)
:- pred lambda(T::in, list(T)::out) is det.
% CHECK-LABEL: pred type_infos.lambda/2 mode 0:
% CHECK-NEXT: type_info arguments: T
% CHECK-NEXT: call pred type_infos.unused/2: T (argument)
:- pred closure(int::out) is det.
% CHECK-LABEL: pred type_infos.closure/1 mode 0:
% CHECK-NEXT: not analysed: Closure
:- pred inferred(int::out) is det.
% CHECK-LABEL: pred type_infos.inferred/1 mode 0:
% CHECK-NEXT: call pred type_infos.q/3: int (constant), string (constant)
% CHECK-NEXT: application func list.length/1: pair.pair(int, string) (constant)
% CHECK-LABEL: pred type_infos.unused/2 mode 0:
% CHECK-NEXT: type_info arguments: T
% CHECK-LABEL: pred type_infos.ret/1 mode 0:
% CHECK-NEXT: type_info arguments: T
% CHECK-LABEL: pred type_infos.q/3 inferred mode 0:
% CHECK-NEXT: type_info arguments: T1, T2
:- implementation.
:- pred unused(T::in, int::out) is det.
unused(_, 1).
:- pred ret(list(T)::out) is det.
ret([]).
two(X, Y, Y - X).
rev(X, Y, X - Y).
f(X, Y) = X - Y.
eqv(X, X).
calls(N) :-
    two(1, "a", P),
    ret(L : list(list(int))),
    unused(P, _),
    unused({1, 'c'}, _),
    unused(f(1, [2]), _),
    eqv(1 - 2, _),
    N = length(L).
poly(X, [X - 1]) :-
    two(X, [X], _),
    unused(X - X, _),
    compare(_, [X], [X]).
undetermined(N) :-
    ret(L),
    unused(L, N).
tests(X, L, N) :-
    ( if X = Y, L = M, Y = X, M = L then N = 1 else N = 0 ),
    Y = X, M = L.
fields(B0, X, B) :-
    _ = B0 ^ contents,
    B = B0 ^ contents := X.
lambda(X, L) :-
    P = (pred(Y::in, Z::out) is det :- unused(Y, _), Z = [Y, X]),
    P(X, L).
closure(N) :-
    unused(eqv, N).
q(X, Y, Z) :-
    Z = [X - Y].
inferred(N) :-
    q(1, "a", L),
    N = length(L).
