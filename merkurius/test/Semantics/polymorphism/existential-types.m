% Where the type_infos and typeclass_infos of existential types come from,
% as in mmc. An existentially typed predicate's or function's hidden
% arguments are those of its universally quantified type variables and
% constraints, then outputs: a type_info for each existentially quantified
% type variable that no existential constraint has, in the order of their
% first appearance, and then a typeclass_info for each existential
% constraint, in theirs. Each clause gives them for the types it chose, as
% a call passes them: a constant, its own hidden argument, a cell, or what
% a goal gave. A caller gets them from the call, and takes the type_info of
% a constrained type out of the typeclass_info it got. A construction with
% an existentially typed constructor, 'new f', passes those of its
% existentially quantified type variables, in their declared order, and
% its constraints; a deconstruction gets them, in a switch as elsewhere.
%
% RUN: merkc -fc1 -fdebug-dump-polymorphism %s 2>%t.err | FileCheck %s
% RUN: %if mmc %{ %compare-polymorphism %s %}

:- module existential_types.
:- interface.
:- import_module list.
:- typeclass sh(T) where [ func sh(T) = int ].
:- typeclass sub(T) <= sh(T) where [ func sub(T) = int ].
:- typeclass two(A, B) where [ func two(A, B) = int ].
:- instance sh(int).
:- instance sub(int).
:- instance sh(list(T)) <= sh(T).
:- instance two(int, string).

:- type box ---> some [T] box(T).
:- type cbox ---> some [T] cbox(T) => sub(T).
:- type pairs(U) ---> some [T] pairs(U, T, list(T)).
:- type k
    --->    some [A, B] k(B, A, B)
    ;       some [C] j(C) => sh(C)
    ;       some [D, E] m(E, D) => two(D, E).
:- type ab ---> a ; b.

:- some [T] pred out(int::in, T::out) is det.
:- some [T] func fout(int) = T.
:- some [T] pred cout(U::in, T::out) is det => sh(T).
:- some [T, V] pred two(U::in, T::out, V::out, U::out) is det => sh(V).
:- all [U] some [T] ((pred wrap(U::in, list(T)::out) is det => sh(T))
    <= sh(U)).
:- all [U, W] some [T, V] ((pred mix(T::out, W::in, V::out, U::in) is det
    => (two(T, V), sh(T))) <= sh(W)).
:- some [T, V] pred order(V::out, T::out) is det.
:- some [T] pred bound(U::in, T::out) is det.
:- some [T] pred pass(int::in, T::out) is det => sh(T).
:- some [T] pred clauses(ab::in, T::out) is det.

:- pred calls(int::out) is det.
:- pred given(int::out) is det.
:- pred boxes(box::in, cbox::in, pairs(int)::in, int::out) is det.
:- pred cases(k::in, int::out) is det.
:- pred make(int::in, box::out, cbox::out, pairs(string)::out, k::out,
    k::out, k::out) is det.
:- pred unused(T::in, int::out) is det.
:- pred use(T::in, int::out) is det <= sh(T).
:- pred use2(A::in, B::in, int::out) is det <= two(A, B).

:- implementation.
:- instance sh(int) where [ sh(X) = X ].
:- instance sub(int) where [ sub(X) = X ].
:- instance sh(list(T)) <= sh(T) where [ sh(_) = 0 ].
:- instance two(int, string) where [ two(_, _) = 2 ].

unused(_, 1).
use(X, sh(X)).
use2(A, B, two(A, B)).

% CHECK-LABEL: pred existential_types.out/2 mode 0:
% CHECK-NEXT: hidden arguments: T (output)
% CHECK-NEXT: results: int (constant)
out(N, N).

% CHECK-LABEL: func existential_types.fout/1 mode 0:
% CHECK-NEXT: hidden arguments: T (output)
% CHECK-NEXT: results: list.list(int) (constant)
fout(N) = [N].

% CHECK-LABEL: pred existential_types.cout/2 mode 0:
% CHECK-NEXT: hidden arguments: U, existential_types.sh(T) (output)
% CHECK-NEXT: results: existential_types.sh(int) (constant)
cout(_, 3).

% CHECK-LABEL: pred existential_types.two/4 mode 0:
% CHECK-NEXT: hidden arguments: U, T (output), existential_types.sh(V) (output)
% CHECK-NEXT: results: int (constant), existential_types.sh(list.list(int)) (constant)
two(U, 1, [2], U).

% CHECK-LABEL: pred existential_types.wrap/2 mode 0:
% CHECK-NEXT: hidden arguments: existential_types.sh(U), existential_types.sh(T) (output)
% CHECK-NEXT: type_infos in typeclass_infos: U (type_info 1 of existential_types.sh(U))
% CHECK-NEXT: results: existential_types.sh(list.list(U)) (instance with existential_types.sh(U) (argument))
wrap(X, [[X]]).

% The universal constraint's typeclass_info comes before the existential
% ones, whatever the order of their variables.
% CHECK-LABEL: pred existential_types.mix/4 mode 0:
% CHECK-NEXT: hidden arguments: U, existential_types.sh(W), existential_types.two(T, V) (output), existential_types.sh(T) (output)
% CHECK-NEXT: type_infos in typeclass_infos: W (type_info 1 of existential_types.sh(W))
% CHECK-NEXT: results: existential_types.two(int, string) (constant), existential_types.sh(int) (constant)
mix(1, _, "s", _).

% CHECK-LABEL: pred existential_types.order/2 mode 0:
% CHECK-NEXT: hidden arguments: V (output), T (output)
% CHECK-NEXT: results: int (constant), string (constant)
order(1, "a").

% CHECK-LABEL: pred existential_types.bound/2 mode 0:
% CHECK-NEXT: hidden arguments: U, T (output)
% CHECK-NEXT: results: U (argument)
bound(U, U).

% The type the clause gives T is the one the call gave, which is T.
% CHECK-LABEL: pred existential_types.pass/2 mode 0:
% CHECK-NEXT: hidden arguments: existential_types.sh(T) (output)
% CHECK-NEXT: call pred existential_types.cout/2: int (constant), existential_types.sh(T) (output)
% CHECK-NEXT: results: existential_types.sh(T) (output of existential_types.cout)
pass(N, T) :- cout(N, T).

% Each clause gives the same type.
% CHECK-LABEL: pred existential_types.clauses/2 mode 0:
% CHECK-NEXT: hidden arguments: T (output)
% CHECK-NEXT: results: int (constant)
% CHECK-NEXT: results: int (constant)
clauses(a, 1).
clauses(b, 2).

% CHECK-LABEL: pred existential_types.calls/1 mode 0:
% CHECK-NEXT: call pred existential_types.out/2: E[[OUT:[0-9]+]] (output)
% CHECK-NEXT: call pred existential_types.unused/2: E[[OUT]] (output of existential_types.out)
% CHECK-NEXT: application func existential_types.fout/1: E[[FOUT:[0-9]+]] (output)
% CHECK-NEXT: call pred existential_types.unused/2: E[[FOUT]] (output of existential_types.fout)
% CHECK-NEXT: call pred existential_types.cout/2: string (constant), existential_types.sh(E[[C:[0-9]+]]) (output)
% CHECK-NEXT: application func existential_types.sh/1: existential_types.sh(E[[C]]) (output of existential_types.cout)
% CHECK-NEXT: call pred existential_types.unused/2: list.list(E[[C]]) (cell)
% CHECK-NEXT: call pred existential_types.two/4: int (constant), E[[D:[0-9]+]] (output), existential_types.sh(E[[E:[0-9]+]]) (output)
% CHECK-NEXT: call pred existential_types.unused/2: E[[D]] (output of existential_types.two)
% CHECK-NEXT: application func existential_types.sh/1: existential_types.sh(E[[E]]) (output of existential_types.two)
% CHECK-NEXT: call pred existential_types.wrap/2: existential_types.sh(list.list(int)) (constant), existential_types.sh(E[[W:[0-9]+]]) (output)
% CHECK-NEXT: call pred existential_types.unused/2: list.list(E[[W]]) (cell)
calls(N) :-
    out(1, A), unused(A, _),
    B = fout(2), unused(B, _),
    cout("a", C), _ = sh(C), unused([C], _),
    two(1, D, E, _), unused(D, _), _ = sh(E),
    wrap([4], W), unused(W, N).

% What a goal gave satisfies a superclass's constraint, and an instance's.
% CHECK-LABEL: pred existential_types.given/1 mode 0:
% CHECK-NEXT: call pred existential_types.pass/2: existential_types.sh(E[[P:[0-9]+]]) (output)
% CHECK-NEXT: call pred existential_types.use/2: existential_types.sh(list.list(E[[P]])) (instance with existential_types.sh(E[[P]]) (output of existential_types.pass))
% CHECK-NEXT: construct existential_types.'new cbox'/1: existential_types.sub(int) (constant)
% CHECK-NEXT: deconstruct existential_types.cbox/1: existential_types.sub(E[[S:[0-9]+]]) (output)
% CHECK-NEXT: call pred existential_types.use/2: existential_types.sh(E[[S]]) (superclass 1 of existential_types.sub(E[[S]]) (output of existential_types.cbox))
% CHECK-NEXT: call pred existential_types.unused/2: E[[S]] (type_info 2 of existential_types.sub(E[[S]]))
given(N) :-
    pass(1, P), use([P], _),
    'new cbox'(3) = cbox(S), use(S, N), unused(S, _).

% CHECK-LABEL: pred existential_types.boxes/4 mode 0:
% CHECK-NEXT: deconstruct existential_types.box/1: E[[X:[0-9]+]] (output)
% CHECK-NEXT: deconstruct existential_types.cbox/1: existential_types.sub(E[[Y:[0-9]+]]) (output)
% CHECK-NEXT: deconstruct existential_types.pairs/3: E[[T:[0-9]+]] (output)
% CHECK-NEXT: call pred existential_types.unused/2: E[[X]] (output of existential_types.box)
% CHECK-NEXT: application func existential_types.sub/1: existential_types.sub(E[[Y]]) (output of existential_types.cbox)
% CHECK-NEXT: call pred existential_types.unused/2: int (constant)
% CHECK-NEXT: call pred existential_types.unused/2: E[[T]] (output of existential_types.pairs)
% CHECK-NEXT: call pred existential_types.unused/2: list.list(E[[T]]) (cell)
boxes(box(X), cbox(Y), pairs(U, T, L), N) :-
    unused(X, _), _ = sub(Y), unused(U, _), unused(T, _), unused(L, N).

% CHECK-LABEL: pred existential_types.cases/2 mode 0:
% CHECK: deconstruct existential_types.j/1: existential_types.sh(E[[J:[0-9]+]]) (output)
% CHECK-NEXT: application func existential_types.sh/1: existential_types.sh(E[[J]]) (output of existential_types.j)
% CHECK-NEXT: deconstruct existential_types.k/3: E[[A:[0-9]+]] (output), E[[B:[0-9]+]] (output)
% CHECK-NEXT: call pred existential_types.unused/2: E[[B]] (output of existential_types.k)
% CHECK-NEXT: call pred existential_types.unused/2: E[[A]] (output of existential_types.k)
% CHECK-NEXT: deconstruct existential_types.m/2: existential_types.two(E[[D:[0-9]+]], E[[E:[0-9]+]]) (output)
% CHECK-NEXT: call pred existential_types.use2/3: existential_types.two(E[[D]], E[[E]]) (output of existential_types.m)
cases(k(X, Y, _), N) :- unused(X, _), unused(Y, N).
cases(j(X), sh(X)).
cases(m(E, D), N) :- use2(D, E, N).

% CHECK-LABEL: pred existential_types.make/7 mode 0:
% CHECK-NEXT: construct existential_types.'new box'/1: int (constant)
% CHECK-NEXT: construct existential_types.'new cbox'/1: existential_types.sub(int) (constant)
% CHECK-NEXT: construct existential_types.'new pairs'/3: int (constant)
% CHECK-NEXT: construct existential_types.'new k'/3: string (constant), int (constant)
% CHECK-NEXT: construct existential_types.'new j'/1: existential_types.sh(list.list(int)) (constant)
% CHECK-NEXT: construct existential_types.'new m'/2: existential_types.two(int, string) (constant)
make(N, 'new box'(N), 'new cbox'(N), 'new pairs'("a", N, []),
    'new k'(N, "a", N), 'new j'([N]), 'new m'("s", N)).
