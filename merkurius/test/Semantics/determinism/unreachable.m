% Goals after one with no solutions, as mmc treats them. Its mode analysis
% leaves out the goals after a call declared failure or erroneous, or a
% unification that cannot succeed, and the then part of an if-then-else
% whose condition is so; their errors are not reported, though their
% variables still count as used until mmc quantifies the body again. An
% if-then-else whose condition has no solutions is its else part, where
% the manual counts the then part too (see the design document's
% register). Only the first solution is wanted of the goals before one
% that mode analysis finds erroneous, as nothing after it asks for
% another. In a declaration's
% notes, the test of a switch's variable against its case's constant is
% true, as mmc makes it.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err | FileCheck %s
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: --implicit-check-not=because: %s < %t.err
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module unreachable.
:- interface.
:- import_module list.
:- type t ---> a ; b ; c(int).

:- pred cond_fails(int::in, int::out) is det.
:- pred cond_erroneous(int::in, int::out) is det.
% ERR-DAG: unreachable.m:[[@LINE+1]]:9: error: the determinism declaration of cond_fails_else(in, out) is not satisfied: it is declared det, but inferred semidet
:- pred cond_fails_else(int::in, int::out) is det.
% ERR-DAG: unreachable.m:[[@LINE+1]]:9: error: the determinism declaration of cond_may_fail(in, out) is not satisfied: it is declared det, but inferred semidet
:- pred cond_may_fail(int::in, int::out) is det.
:- pred then_cc(int::in, int::out) is det.
:- pred then_require(int::in, int::out) is det.
:- pred inferred_cond(int::in, int::out) is det.
:- pred before_failure(int::in, int::out) is failure.
:- pred before_erroneous(int::in, int::out) is erroneous.
:- pred before_inferred(int::in, int::out) is erroneous.
:- pred dropped_use(list(int)::in, int::out) is nondet.
:- pred dropped_unused(list(int)::in, int::out) is nondet.
:- pred unify_dropped_use(list(int)::in, t::in(bound(a)), int::out) is nondet.
:- pred unify_dropped_unused(list(int)::in, t::in(bound(a)), int::out) is nondet.
% ERR-DAG: unreachable.m:[[@LINE+1]]:9: error: the determinism declaration of case_test(in, in(bound(a ; b)), out) is not satisfied: it is declared erroneous, but inferred semidet
:- pred case_test(t::in, t::in(bound(a ; b)), int::out) is erroneous.

:- implementation.
:- import_module int, require.

:- pred fl(int::in, int::out) is failure.
fl(_, _) :- fail.
:- pred er(int::in, int::out) is erroneous.
er(_, _) :- error("er").
:- pred cm(int::in, int::out) is cc_multi.
cm(X, Y) :- ( Y = X ; Y = X + 1 ).
:- pred sd(int::in, int::out) is semidet.
sd(X, X) :- X > 0.
:- pred nd(int::in, int::out) is nondet.
nd(X, Y) :- X > 0, ( Y = X ; Y = 1 ).

% A condition that cannot succeed leaves out the then part, so the
% if-then-else is det; one that is erroneous makes it erroneous.
% CHECK-LABEL: cond_fails(in, out) is det, inferred det
% CHECK: if-then-else [det]
% CHECK-NEXT: if
% CHECK-NEXT: call pred unreachable.fl/2 mode 0: fl(X, Z) [failure]
% CHECK-NEXT: then
% CHECK-NEXT: true [det]
cond_fails(X, Y) :- ( if fl(X, Z) then nd(Z, Y) else Y = X ).
% CHECK-LABEL: cond_erroneous(in, out) is det, inferred erroneous
cond_erroneous(X, Y) :- ( if er(X, Z) then nd(Z, Y) else Y = X ).
% The reasons are then the else part's.
% ERR-DAG: unreachable.m:[[@LINE+1]]:59: because: the call to sd(in, out) can fail
cond_fails_else(X, Y) :- ( if fl(X, Z) then nd(Z, Y) else sd(X, Y) ).
% ERR-DAG: unreachable.m:[[@LINE+1]]:64: because: the call to sd(in, out) can fail
cond_may_fail(X, Y) :- ( if X > 0, fl(X, Z) then nd(Z, Y) else sd(X, Y) ).

% No errors in a then part left out.
then_cc(X, Y) :- ( if fl(X, Z) then cm(Z, Y) else Y = X ).
then_require(X, Y) :- ( if fl(X, Z) then require_det sd(Z, Y) else Y = X ).

% A condition that cannot succeed by inference only, after mode analysis,
% keeps the then part, though the if-then-else is its else part.
:- pred ufl(int::in, int::out).
ufl(X, Y) :- fl(X, Y).
% CHECK-LABEL: inferred_cond(in, out) is det, inferred det
% CHECK: call pred unreachable.nd/2 mode 0: nd(Z, Y) [nondet]
inferred_cond(X, Y) :- ( if ufl(X, Z) then nd(Z, Y) else Y = X ).
:- pred inferred_then_cc(int::in, int::out).
% ERR-DAG: unreachable.m:[[@LINE+1]]:47: error: the call to cm(in, out) with determinism cc_multi is in a context that requires all solutions
inferred_then_cc(X, Y) :- ( if ufl(X, Z) then cm(Z, Y) else Y = X ).

% A committed choice call before a goal that can fail is an error; before
% an erroneous one, only its first solution is wanted.
% ERR-DAG: unreachable.m:[[@LINE+2]]:25: error: the call to cm(in, out) with determinism cc_multi is in a context that requires all solutions
% ERR-DAG: unreachable.m:[[@LINE+1]]:35: because: the call to fl(in, out) can fail
before_failure(X, Y) :- cm(X, Z), fl(Z, Y).
% CHECK-LABEL: before_erroneous(in, out) is erroneous, inferred erroneous
% CHECK: call pred unreachable.cm/2 mode 0: cm(X, Z) [cc_multi]
before_erroneous(X, Y) :- cm(X, Z), er(Z, Y).
% But mmc finds that by mode analysis, so not before a goal erroneous only
% by inference.
:- pred uer(int::in, int::out).
uer(X, Y) :- er(X, Y).
% ERR-DAG: unreachable.m:[[@LINE+1]]:26: error: the call to cm(in, out) with determinism cc_multi is in a context that requires all solutions
before_inferred(X, Y) :- cm(X, Z), uer(Z, Y).

% The variables of the goals left out after a call that cannot succeed,
% or a unification, count as used: M, used in Y = M, is an output of the
% committed choice call, which is then an error, where _ is not.
% ERR-DAG: unreachable.m:[[@LINE+2]]:40: error: the call to cm(in, out) with determinism cc_multi is in a context that requires all solutions
% ERR-DAG: unreachable.m:[[@LINE+1]]:50: because: the call to fl(in, out) can fail
dropped_use(L, Y) :- ( L = [], Y = 0 ; cm(1, M), fl(1, _), Y = M ).
dropped_unused(L, Y) :- ( L = [], Y = 0 ; cm(1, _), fl(1, _), Y = 3 ).
% ERR-DAG: unreachable.m:[[@LINE+1]]:49: error: the call to cm(in, out) with determinism cc_multi is in a context that requires all solutions
unify_dropped_use(L, T, Y) :- ( L = [], Y = 0 ; cm(1, M), T = b, Y = M ).
unify_dropped_unused(L, T, Y) :- ( L = [], Y = 0 ; cm(1, _), T = b, Y = 3 ).

% The second argument cannot be c(_), so the third clause is its first
% argument's test and fail, which has no note; the c/1 test has one.
% ERR-DAG: unreachable.m:[[@LINE+4]]:1: because: the switch on HeadVar__1 does not cover b/0
% ERR-DAG: unreachable.m:[[@LINE+3]]:11: because: in the case c/1 of the switch on HeadVar__1: the unification of HeadVar__1 and c/1 can succeed
% ERR-DAG: unreachable.m:[[@LINE+2]]:17: because: in the case c/1 of the switch on HeadVar__1: the unification of HeadVar__2 and a/0 can fail and can succeed
% ERR-DAG: unreachable.m:[[@LINE+1]]:13: because: in the case c/1 of the switch on HeadVar__1: the unification of HeadVar__3 and K can succeed
case_test(c(K), a, K).
case_test(a, c(_), 0).
