% Where a procedure's typeclass_infos come from, as in mmc. Its hidden
% arguments are a type_info for each type variable no constraint has, and
% then a typeclass_info for each constraint, in their orders; an instance
% method's are the instance's and then the method's. A constrained type
% variable's type_info is in the typeclass_info of the first constraint
% that has it, after its type class's superclasses. A call passes, for each
% constraint of its callee, the caller's own typeclass_info, a superclass's
% taken out of one, through as many superclasses as it takes, a constant
% for a constraint without type variables, or a cell of the instance's
% with the typeclass_infos of the instance's constraints and of its type
% class's superclasses on its types. A method call passes its type class's
% typeclass_info first.
%
% RUN: merkc -fc1 -fdebug-dump-polymorphism %s 2>%t.err | FileCheck %s
% RUN: %if mmc %{ %compare-polymorphism %s %}

:- module typeclass_infos.
:- interface.
:- import_module list, maybe, pair.
:- typeclass sh(T) where [ func sh(T) = int ].
:- typeclass eq(T) where [ pred eq(T::in, T::in) is semidet ].
:- typeclass sub(T) <= (eq(T), sh(T)) where [ func sub(T) = int ].
:- typeclass subsub(T) <= sub(T) where [ ].
:- typeclass two(A, B) where [ pred two(A::in, B::in) is det ].
:- typeclass c(T) where [
    pred m(T::in, U::in, V::in, int::out) is det <= sh(V)
].
:- instance sh(int).
:- instance eq(int).
:- instance sub(int).
:- instance sh(list(T)) <= sh(T).
:- instance eq(list(T)) <= eq(T).
:- instance sub(list(T)) <= sub(T).
:- instance sh(pair(A, B)) <= (sh(B), sh(A)).
:- instance two(int, string).
:- instance two(list(A), maybe(B)) <= sh(B).
:- instance c(list(T)) <= sh(T).

:- pred use(T::in, int::out) is det <= sh(T).
% CHECK-LABEL: pred typeclass_infos.use/2 mode 0:
% CHECK-NEXT: hidden arguments: typeclass_infos.sh(T)
% CHECK-NEXT: type_infos in typeclass_infos: T (type_info 1 of typeclass_infos.sh(T))
% CHECK-NEXT: application func typeclass_infos.sh/1: typeclass_infos.sh(T) (argument)
:- pred mixed(T::in, U::in, V::in, int::out) is det <= (sh(V), sh(U)).
% CHECK-LABEL: pred typeclass_infos.mixed/4 mode 0:
% CHECK-NEXT: hidden arguments: T, typeclass_infos.sh(V), typeclass_infos.sh(U)
% CHECK-NEXT: type_infos in typeclass_infos: V (type_info 1 of typeclass_infos.sh(V)), U (type_info 1 of typeclass_infos.sh(U))
% CHECK-NEXT: call pred typeclass_infos.unused/2: T (argument)
% CHECK-NEXT: call pred typeclass_infos.unused/2: U (type_info 1 of typeclass_infos.sh(U))
% CHECK-NEXT: call pred typeclass_infos.use/2: typeclass_infos.sh(V) (argument)
:- pred chain(T::in, int::out) is det <= subsub(T).
% CHECK-LABEL: pred typeclass_infos.chain/2 mode 0:
% CHECK-NEXT: hidden arguments: typeclass_infos.subsub(T)
% CHECK-NEXT: type_infos in typeclass_infos: T (type_info 2 of typeclass_infos.subsub(T))
% CHECK-NEXT: application func typeclass_infos.sh/1: typeclass_infos.sh(T) (superclass 2 of typeclass_infos.sub(T) (superclass 1 of typeclass_infos.subsub(T) (argument)))
% CHECK-NEXT: call pred typeclass_infos.eq/2: typeclass_infos.eq(T) (superclass 1 of typeclass_infos.sub(T) (superclass 1 of typeclass_infos.subsub(T) (argument)))
% CHECK-NEXT: call pred typeclass_infos.unused/2: T (type_info 2 of typeclass_infos.subsub(T))
:- pred both(T::in, int::out) is det <= (sh(T), sub(T)).
% CHECK-LABEL: pred typeclass_infos.both/2 mode 0:
% CHECK-NEXT: hidden arguments: typeclass_infos.sh(T), typeclass_infos.sub(T)
% CHECK-NEXT: type_infos in typeclass_infos: T (type_info 1 of typeclass_infos.sh(T))
% CHECK-NEXT: application func typeclass_infos.sh/1: typeclass_infos.sh(T) (argument)
:- pred calls(int::out) is det.
% CHECK-LABEL: pred typeclass_infos.calls/1 mode 0:
% CHECK-NEXT: call pred typeclass_infos.use/2: typeclass_infos.sh(int) (constant)
% CHECK-NEXT: call pred typeclass_infos.use/2: typeclass_infos.sh(list.list(list.list(int))) (constant)
% CHECK-NEXT: call pred typeclass_infos.mixed/4: string (constant), typeclass_infos.sh(int) (constant), typeclass_infos.sh(list.list(int)) (constant)
% CHECK-NEXT: call pred typeclass_infos.tw/2: typeclass_infos.two(int, string) (constant)
:- pred instances(A::in, B::in, int::out) is det <= (sh(A), sub(B)).
% CHECK-LABEL: pred typeclass_infos.instances/3 mode 0:
% CHECK-NEXT: hidden arguments: typeclass_infos.sh(A), typeclass_infos.sub(B)
% CHECK-NEXT: type_infos in typeclass_infos: A (type_info 1 of typeclass_infos.sh(A)), B (type_info 3 of typeclass_infos.sub(B))
% CHECK-NEXT: application func typeclass_infos.sh/1: typeclass_infos.sh(pair.pair(A, B)) (instance with typeclass_infos.sh(B) (superclass 2 of typeclass_infos.sub(B) (argument)), typeclass_infos.sh(A) (argument))
% CHECK-NEXT: call pred typeclass_infos.tw/2: typeclass_infos.two(list.list(A), maybe.maybe(B)) (instance with typeclass_infos.sh(B) (superclass 2 of typeclass_infos.sub(B) (argument)))
% CHECK-NEXT: call pred typeclass_infos.subs/2: typeclass_infos.sub(list.list(B)) (instance with typeclass_infos.sub(B) (argument), typeclass_infos.eq(list.list(B)) (instance with typeclass_infos.eq(B) (superclass 1 of typeclass_infos.sub(B) (argument))), typeclass_infos.sh(list.list(B)) (instance with typeclass_infos.sh(B) (superclass 2 of typeclass_infos.sub(B) (argument))))
% CHECK-NEXT: call pred typeclass_infos.unused/2: pair.pair(A, B) (cell)
:- pred methods(T::in, U::in, V::in, int::out) is det <= (c(T), sh(V)).
% CHECK-LABEL: pred typeclass_infos.methods/4 mode 0:
% CHECK-NEXT: hidden arguments: U, typeclass_infos.c(T), typeclass_infos.sh(V)
% CHECK-NEXT: type_infos in typeclass_infos: T (type_info 1 of typeclass_infos.c(T)), V (type_info 1 of typeclass_infos.sh(V))
% CHECK-NEXT: call pred typeclass_infos.m/4: U (argument), typeclass_infos.c(T) (argument), typeclass_infos.sh(V) (argument)
% CHECK-NEXT: call pred typeclass_infos.m/4: U (argument), typeclass_infos.c(list.list(int)) (constant), typeclass_infos.sh(int) (constant)
:- pred tw(A::in, B::in) is det <= two(A, B).
% CHECK-LABEL: pred typeclass_infos.tw/2 mode 0:
% CHECK-NEXT: hidden arguments: typeclass_infos.two(A, B)
% CHECK-NEXT: type_infos in typeclass_infos: A (type_info 1 of typeclass_infos.two(A, B)), B (type_info 2 of typeclass_infos.two(A, B))
% CHECK-NEXT: call pred typeclass_infos.two/2: typeclass_infos.two(A, B) (argument)
:- pred subs(T::in, int::out) is det <= sub(T).
:- pred unused(T::in, int::out) is det.

:- implementation.
:- import_module int.
:- instance sh(int) where [ sh(X) = X ].
:- instance eq(int) where [ eq(X, X) ].
:- instance sub(int) where [ sub(X) = X ].
:- instance sh(list(T)) <= sh(T) where [
    sh(L) = ( if L = [H | _] then sh(H) else 0 )
].
:- instance eq(list(T)) <= eq(T) where [ eq(_, _) :- true ].
:- instance sub(list(T)) <= sub(T) where [ sub(L) = sh(L) ].
:- instance sh(pair(A, B)) <= (sh(B), sh(A)) where [ sh(A - _) = sh(A) ].
:- instance two(int, string) where [ two(_, _) ].
:- instance two(list(A), maybe(B)) <= sh(B) where [
    (two(_, M) :- ( if M = yes(X) then _ = sh(X) else true ))
].
:- instance c(list(T)) <= sh(T) where [
    (m(L, _, Z, N) :- unused(L, _), N = sh(Z))
].
% CHECK-LABEL: method typeclass_infos.sh(list.list(T)) func typeclass_infos.sh/1 mode 0
% CHECK-NEXT: hidden arguments: typeclass_infos.sh(T)
% CHECK-NEXT: type_infos in typeclass_infos: T (type_info 1 of typeclass_infos.sh(T))
% CHECK-NEXT: application func typeclass_infos.sh/1: typeclass_infos.sh(T) (argument)
% CHECK-LABEL: method typeclass_infos.sub(list.list(T)) func typeclass_infos.sub/1 mode 0
% CHECK-NEXT: hidden arguments: typeclass_infos.sub(T)
% CHECK-NEXT: type_infos in typeclass_infos: T (type_info 3 of typeclass_infos.sub(T))
% CHECK-NEXT: application func typeclass_infos.sh/1: typeclass_infos.sh(list.list(T)) (instance with typeclass_infos.sh(T) (superclass 2 of typeclass_infos.sub(T) (argument)))
% CHECK-LABEL: method typeclass_infos.sh(pair.pair(A, B)) func typeclass_infos.sh/1 mode 0
% CHECK-NEXT: hidden arguments: typeclass_infos.sh(B), typeclass_infos.sh(A)
% CHECK-LABEL: method typeclass_infos.two(list.list(A), maybe.maybe(B)) pred typeclass_infos.two/2 mode 0
% CHECK-NEXT: hidden arguments: A, typeclass_infos.sh(B)
% CHECK-LABEL: method typeclass_infos.c(list.list(T)) pred typeclass_infos.m/4 mode 0
% CHECK-NEXT: hidden arguments: typeclass_infos.sh(T), U, typeclass_infos.sh(V)
% CHECK-NEXT: type_infos in typeclass_infos: T (type_info 1 of typeclass_infos.sh(T)), V (type_info 1 of typeclass_infos.sh(V))
% CHECK-NEXT: call pred typeclass_infos.unused/2: list.list(T) (cell)
% CHECK-NEXT: application func typeclass_infos.sh/1: typeclass_infos.sh(V) (argument)
unused(_, 1).
use(X, sh(X)).
mixed(X, Y, Z, N) :- unused(X, _), unused(Y, _), use(Z, N).
chain(X, N) :- N = sh(X), ( if eq(X, X) then true else true ), unused(X, _).
both(X, N) :- N = sh(X).
calls(N) :-
    use(1, A), use([[2]], B), mixed("a", [3], 4, C), tw(1, "b"),
    N = A + B + C.
instances(A, B, N) :-
    N = sh(A - B), tw([A], yes(B)), subs([B], _), unused(A - B, _).
methods(X, Y, Z, N) :- m(X, Y, Z, N), m([1], Y, 1, _).
tw(A, B) :- two(A, B).
subs(X, sub(X)).
