% Mode inference (Overton, "Precise and Expressive Mode Systems for Typed
% Logic Programming Languages", 2003, section 3.4.2; reference manual,
% "Modes"): a call of a predicate without a `:- pred` declaration makes a
% procedure for it, from the call's argument insts, normalised, with the
% final insts of its clauses. As in mmc, a unique argument not used after
% the call is clobbered, di; one used after is (unique >> ground) when the
% predicate shares it; a call whose insts an inferred mode matches, also
% as an implied mode, reuses it, and one that it would clobber gets a
% new one; a recursive predicate's modes are a fixpoint, and so are those
% of predicates that call each other; a partially instantiated argument
% is not normalised; a call's goals after it get the inferred final
% insts; a call whose mode is rejected waits for other insts, which make
% another mode; and a function without a declaration has the default
% mode.
%
% RUN: merkc -fc1 -fdebug-dump-modes %s | FileCheck %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module inference.
:- interface.
:- import_module list.

:- pred input(int::in, int::out) is det.
:- pred mutual(int::in, int::out) is det.
:- pred dead(list(int)::out) is det.
:- pred live(list(int)::out, list(int)::out) is det.
:- pred implied(int::out) is semidet.
:- pred recursive(list(int)::in, int::out) is det.
:- pred function(int::out) is det.
:- pred partial(list(int)::out) is semidet.
:- pred after(int::out) is det.
:- pred waits(int::out) is det.

:- implementation.
:- import_module int.

:- pred use(int::in) is det.
use(_).

input(X, Y) :-
    add_one(X, Y).

mutual(X, Y) :-
    even(X, Y).

dead(Y) :-
    X = [1],
    id(X, Y).

live(Y, X) :-
    X = [1],
    id(X, Y).

implied(Y) :-
    inc(1, Y),
    inc(2, 3).

recursive(L, N) :-
    len(L, N).

function(Y) :-
    Y = twice(3).
% The default mode, made with the declared procedures.
% CHECK-LABEL: func inference.twice/1 mode 0: twice(in) = out is det

partial(L) :-
    L = [_ | _],
    fill(L),
    L = [_].

after(Z) :-
    q(1, Y),
    r(Z, Y).

waits(Y) :-
    bad(Y),
    Y = 3.

add_one(X, Y) :-
    Y = X + 1.
% CHECK-LABEL: pred inference.add_one/2 inferred mode 0: add_one(in, uo)

even(X, Y) :-
    ( if X = 0 then Y = 0 else odd(X - 1, Y) ).
% CHECK-LABEL: pred inference.even/2 inferred mode 0: even(in, uo)

odd(X, Y) :-
    ( if X = 0 then Y = 1 else even(X - 1, Y) ).
% CHECK-LABEL: pred inference.odd/2 inferred mode 0: odd(di, uo)

id(X, X).
% CHECK-LABEL: pred inference.id/2 inferred mode 0: id(di, uo)
% CHECK-LABEL: pred inference.id/2 inferred mode 1: id((unique >> ground), out)

inc(X, Y) :-
    Y = X + 1.
% CHECK-LABEL: pred inference.inc/2 inferred mode 0: inc(di, uo)
% CHECK-NOT: inferred mode 1: inc

len([], 0).
len([_ | T], N) :-
    len(T, M),
    N = M + 1.
% CHECK-LABEL: pred inference.len/2 inferred mode 0: len(in, uo)

twice(X) = X * 2.

fill([1 | _]).
% CHECK-LABEL: pred inference.fill/1 inferred mode 0: fill((unique('[|]'(free, free)) >> unique('[|]'(unique(1), free))))

q(X, Y) :-
    Y = X + 1.
% CHECK-LABEL: pred inference.q/2 inferred mode 0: q(di, uo)

r(Z, Y) :-
    Z = Y + 1.
% CHECK-LABEL: pred inference.r/2 inferred mode 0: r(uo, di)

bad(Y) :-
    use(Y).
% CHECK-LABEL: pred inference.bad/1 inferred mode 0: bad(out(not_reached))
% CHECK-NEXT: mode error
% CHECK-LABEL: pred inference.bad/1 inferred mode 1: bad((unique >> ground))
