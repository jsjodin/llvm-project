% Disjunctions, if-then-elses, negations and predicates of several clauses
% that mode analysis accepts, and how it runs them (Overton, "Precise and
% Expressive Mode Systems for Typed Logic Programming Languages", 2003,
% chapter 3): each branch from the insts before it, merged after; a
% condition binding what only it and the then part use; a branched goal
% or negation waiting, as a whole, for a goal after it; a branch that is
% not reached; a variable free in one branch but not used after; and the
% clauses of a predicate, each with head unifications, as mmc checks them.
%
% RUN: merkc -fc1 -fdebug-dump-modes %s | FileCheck %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module branches.
:- interface.
:- import_module list.

:- pred both(int::out) is multi.
:- pred condition(list(int)::in, int::out) is det.
:- pred waits(int::in, int::out) is semidet.
:- pred negation(int::in, int::out) is semidet.
:- pred unreached(int::out) is multi.
:- pred unused is multi.
:- pred clauses(int::in, int::out) is multi.

:- implementation.

:- pred q(int::(free >> free)) is det.
q(_).

both(X) :-
    ( X = 1
    ; X = 2
    ).
% CHECK-LABEL: pred branches.both/1 mode 0: both(out) is multi
% CHECK: disjunction
% CHECK-NEXT: branch
% CHECK-NEXT: construct X = 1
% CHECK-NEXT: branch
% CHECK-NEXT: construct X = 2
% CHECK-NEXT: final: X unique(1 ; 2)

condition(L, X) :-
    ( if L = [H | _] then
        X = H
    else
        X = 0
    ).
% CHECK-LABEL: pred branches.condition/2 mode 0: condition(in, out) is det
% CHECK: if-then-else
% CHECK-NEXT: if
% CHECK-NEXT: deconstruct L = '[|]'(H, V_{{[0-9]+}})
% CHECK-NEXT: then
% CHECK-NEXT: assign X := H
% CHECK-NEXT: else
% CHECK-NEXT: construct X = 0
% CHECK-NEXT: final: L ground, X ground

waits(A, X) :-
    ( if Y = 1 then
        X = 1
    else
        X = 2
    ),
    Y = A.
% CHECK-LABEL: pred branches.waits/2 mode 0: waits(in, out) is semidet
% CHECK: assign Y := A
% CHECK-NEXT: if-then-else
% CHECK-NEXT: if
% CHECK-NEXT: test Y = 1

negation(A, X) :-
    not X = 1,
    X = A.
% CHECK-LABEL: pred branches.negation/2 mode 0: negation(in, out) is semidet
% CHECK: assign X := A
% CHECK-NEXT: not
% CHECK-NEXT: test X = 1

unreached(X) :-
    ( X = 1
    ; fail
    ).
% CHECK-LABEL: pred branches.unreached/1 mode 0: unreached(out) is multi
% CHECK: disjunction
% CHECK-NEXT: branch
% CHECK-NEXT: construct X = 1
% CHECK-NEXT: branch
% CHECK-NEXT: fail
% CHECK-NEXT: final: X unique(1)

unused :-
    q(X),
    ( X = 1
    ; true
    ).
% CHECK-LABEL: pred branches.unused/0 mode 0: unused is multi
% CHECK: call pred branches.q/1 mode 0: q(X)
% CHECK-NEXT: disjunction

clauses(A, X) :-
    X = A.
clauses(_, 2).
% CHECK-LABEL: pred branches.clauses/2 mode 0: clauses(in, out) is multi
% CHECK: clause at [[@LINE-4]]:1
% CHECK-NEXT: assign A := HeadVar__1
% CHECK-NEXT: assign X := A
% CHECK-NEXT: assign HeadVar__2 := X
% CHECK-NEXT: final: HeadVar__1 ground, HeadVar__2 ground
% CHECK: clause at [[@LINE-7]]:1
