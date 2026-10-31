% Mode errors in disjunctions, if-then-elses, negations and predicates of
% several clauses (Overton, "Precise and Expressive Mode Systems for Typed
% Logic Programming Languages", 2003, chapter 3): a variable free in some
% branches and bound in others, or partly so, which mmc reports at the
% goal run last, with a note at each branch, but with three or more
% branches only at those where it is not ground, if it is free in all of
% them; an error inside a branch; a negation or a condition that binds a
% variable used outside it; and clauses, which are the branches of a
% disjunction, whose head unifications are reported only if no other goal
% waits.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=because: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module branch_errors.
:- interface.
:- import_module list.

:- pred one_branch(int::out) is multi.
:- pred three_branches(int::out) is multi.
:- pred partial_branch(list(int)::out) is multi.
:- pred else_part(int::in, int::out) is det.
:- pred in_branch(int::in, int::out) is multi.
:- pred condition(int::out) is semidet.
:- pred negation(int::out) is semidet.
:- pred clauses(int::out) is multi.
:- pred three_clauses(int::out) is multi.
:- pred final(int::in, int::out, int::out) is multi.
:- pred head_waits(int::out) is multi.
:- pred unused_partial(list(int)::in) is multi.

:- implementation.

:- pred use(int::in) is det.
use(_).

one_branch(X) :-
    (
        X = 1
    ;
        true
% CHECK: [[@LINE-1]]:9: error: in the clause for one_branch(out): mode mismatch in the disjunction: X is ground in some branches but not others
% CHECK: [[@LINE-4]]:13: because: in this branch, X has inst unique(1)
% CHECK: [[@LINE-3]]:9: because: in this branch, X has inst free
    ).
three_branches(X) :-
    (
        X = 1
    ;
        true
    ;
        X = 2
% CHECK: [[@LINE-1]]:13: error: in the clause for three_branches(out): mode mismatch in the disjunction: X is ground in some branches but not others
% CHECK: [[@LINE-4]]:9: because: in this branch, X has inst free
    ).
partial_branch(L) :-
    (
        L = [_ | _]
    ;
        L = [1]
% CHECK: [[@LINE-1]]:13: error: in the clause for partial_branch(out): mode mismatch in the disjunction: L is ground in some branches but not others
% CHECK: [[@LINE-4]]:13: because: in this branch, L has inst unique('[|]'(free, free))
% CHECK: [[@LINE-3]]:9: because: in this branch, L has inst unique('[|]'(unique(1), unique([])))
    ).
else_part(A, X) :-
    ( if A = 1 then
        X = 2
    else
        true
% CHECK: [[@LINE-1]]:9: error: in the clause for else_part(in, out): mode mismatch in the if-then-else: X is ground in some branches but not others
% CHECK: [[@LINE-4]]:13: because: in this branch, X has inst unique(2)
% CHECK: [[@LINE-3]]:9: because: in this branch, X has inst free
    ).
in_branch(A, X) :-
    (
        X = A
    ;
        use(Z),
% CHECK: [[@LINE-1]]:9: error: in the clause for in_branch(in, out): mode error in argument 1 of the call to pred branch_errors.use/1: Z has inst free, but ground is expected
        X = Z
    ).
condition(X) :-
    ( if X = 1 then
% CHECK: [[@LINE-1]]:14: error: in the clause for condition(out): scope error: X is bound inside the condition of an if-then-else, but occurs outside the condition and the then part: it has inst free, and would get unique(1)
        true
    else
        fail
    ).
negation(X) :-
    not X = 1.
% CHECK: [[@LINE-1]]:13: error: in the clause for negation(out): scope error: X is bound inside a negation, but is not local to it: it has inst free, and would get unique(1)
clauses(1).
clauses(_).
% CHECK: [[@LINE-1]]:9: error: in the clause for clauses(out): mode mismatch in the disjunction: HeadVar__1 is ground in some branches but not others
% CHECK: [[@LINE-3]]:1: because: in this branch, HeadVar__1 has inst unique(1)
% CHECK: [[@LINE-3]]:1: because: in this branch, HeadVar__1 has inst free
three_clauses(1).
three_clauses(_).
three_clauses(2).
% CHECK: [[@LINE-1]]:15: error: in the clause for three_clauses(out): mode mismatch in the disjunction: HeadVar__1 is ground in some branches but not others
% CHECK: [[@LINE-3]]:1: because: in this branch, HeadVar__1 has inst free
final(A, X, _) :-
    use(A),
    X = 1.
final(A,
        X,
        _) :-
    use(A),
    X = 2,
    use(A).
% CHECK: [[@LINE-1]]:5: error: in the clause for final(in, out, out): mode error: argument 3, HeadVar__3, did not get sufficiently instantiated: its final inst is free, but ground is expected
head_waits(1).
head_waits(X) :-
    use(X).
% CHECK: [[@LINE-1]]:5: error: in the clause for head_waits(out): mode error in argument 1 of the call to pred branch_errors.use/1: X has inst free, but ground is expected
% A test of a partial term against a ground one does not bind its free
% parts, so V is partial in the then part, which mmc reports though V is
% not used after.
unused_partial(L) :-
    V = [_ | _],
    ( if L = [] then
        V = [8]
    else
        V = L
% CHECK: [[@LINE-1]]:9: error: in the clause for unused_partial(in): mode mismatch in the if-then-else: V is ground in some branches but not others
% CHECK: [[@LINE-4]]:9: because: in this branch, V has inst bound('[|]'(free, free))
% CHECK: [[@LINE-3]]:9: because: in this branch, V has inst bound('[|]'(ground, ground))
    ).
