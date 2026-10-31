% Determinism declarations that are not satisfied (reference manual,
% "Determinism checking and inference"), each with the goals that make the
% difference (Henderson, Somogyi and Conway, "Determinism analysis in the
% Mercury compiler", 1996, section 5.3), as mmc finds them: calls that can
% fail or succeed more than once, unifications that can fail, a switch
% that does not cover a functor, or a case of it, disjunctions with more
% than one disjunct with solutions, each conjunct, the condition of an
% if-then-else and its two parts, a negation that can succeed, the goals
% of a scope, and with no solutions declared, any goal that can succeed; a
% lambda expression whose body is not as declared; and a function whose
% primary mode is declared with more than one solution.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=because: %s
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module goal_errors.
:- interface.
:- import_module list.

:- type t ---> a ; b ; c(int).

% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of call_fails(in, out) is not satisfied: it is declared det, but inferred semidet
:- pred call_fails(int::in, int::out) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of call_multi(out) is not satisfied: it is declared det, but inferred multi
:- pred call_multi(int::out) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of call_nondet(in, out) is not satisfied: it is declared det, but inferred nondet
:- pred call_nondet(int::in, int::out) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of conjuncts(in, out) is not satisfied: it is declared det, but inferred nondet
:- pred conjuncts(int::in, int::out) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of apply(in, out) is not satisfied: it is declared det, but inferred semidet
:- pred apply(int::in, int::out) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of higher_order(in((pred(out) is multi)), out) is not satisfied: it is declared det, but inferred multi
:- pred higher_order(pred(int)::in(pred(out) is multi), int::out) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of test(in, in) is not satisfied: it is declared det, but inferred semidet
:- pred test(t::in, t::in) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of deconstruct(in, out) is not satisfied: it is declared det, but inferred semidet
:- pred deconstruct(t::in, int::out) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of cover(in, out) is not satisfied: it is declared det, but inferred semidet
:- pred cover(t::in, int::out) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of int_switch(in, out) is not satisfied: it is declared det, but inferred semidet
:- pred int_switch(int::in, int::out) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of nested(in, in, out) is not satisfied: it is declared det, but inferred semidet
:- pred nested(t::in, t::in, int::out) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of clauses(in, out) is not satisfied: it is declared det, but inferred multi
:- pred clauses(t::in, int::out) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of disjuncts(in, out) is not satisfied: it is declared det, but inferred nondet
:- pred disjuncts(int::in, int::out) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of if_then_else(in, out) is not satisfied: it is declared det, but inferred nondet
:- pred if_then_else(int::in, int::out) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of negation(in) is not satisfied: it is declared det, but inferred semidet
:- pred negation(int::in) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of some_scope(in) is not satisfied: it is declared det, but inferred semidet
:- pred some_scope(list(int)::in) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of no_solutions(in, out) is not satisfied: it is declared erroneous, but inferred semidet
:- pred no_solutions(int::in, int::out) is erroneous.
% No reason is given for a procedure that cannot succeed.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: the determinism declaration of failing(out) is not satisfied: it is declared det, but inferred failure
:- pred failing(int::out) is det.
:- pred lambda(pred(int)::out(pred(out) is det)) is det.
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: error: invalid determinism for primary(in) = out: the primary mode of a function cannot be multi
:- func primary(int) = int is multi.

:- implementation.
:- import_module int.

:- pred sd(int::in, int::out) is semidet.
sd(X, X) :- X > 0.
:- pred m(int::out) is multi.
m(1).
m(2).
:- pred nd(int::in, int::out) is nondet.
nd(X, Y) :- ( Y = X ; X > 0, Y = 1 ).
:- func f(int) = int is semidet.
f(X) = X :- X > 0.

% CHECK-DAG: goal-errors.m:[[@LINE+1]]:21: because: the call to sd(in, out) can fail
call_fails(X, Y) :- sd(X, Y).

% CHECK-DAG: goal-errors.m:[[@LINE+1]]:18: because: the call to m(out) can succeed more than once
call_multi(Y) :- m(Y).

% CHECK-DAG: goal-errors.m:[[@LINE+1]]:22: because: the call to nd(in, out) can fail and can succeed more than once
call_nondet(X, Y) :- nd(X, Y).

conjuncts(X, Y) :-
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:3: because: the call to sd(in, out) can fail
  sd(X, Z),
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:3: because: the call to m(out) can succeed more than once
  m(W),
  Y = Z + W.

% CHECK-DAG: goal-errors.m:[[@LINE+1]]:20: because: the call to f(in) = out can fail
apply(X, Y) :- Y = f(X).

% CHECK-DAG: goal-errors.m:[[@LINE+1]]:23: because: the higher-order call can succeed more than once
higher_order(P, Y) :- P(Y).

% CHECK-DAG: goal-errors.m:[[@LINE+1]]:15: because: the unification of X and Y can fail
test(X, Y) :- X = Y.

% CHECK-DAG: goal-errors.m:[[@LINE+1]]:26: because: the unification of X and c/1 can fail
deconstruct(X, Y) :- X = c(Y).

% CHECK-DAG: goal-errors.m:[[@LINE+1]]:1: because: the switch on HeadVar__1 does not cover c/1
cover(a, 1).
cover(b, 2).

% CHECK-DAG: goal-errors.m:[[@LINE+1]]:23: because: the switch on X can fail
int_switch(X, Y) :- ( X = 1, Y = 1 ; X = 2, Y = 2 ).

nested(X, Z, Y) :-
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:14: because: in the case a/0 of the switch on X: the switch on Z does not cover c/1
  ( X = a, ( Z = a, Y = 1 ; Z = b, Y = 2 )
  ; X = b, Y = 3
  ; X = c(_), Y = 4 ).

clauses(b, 2).
clauses(c(_), 3).
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:9: because: in the case a/0 of the switch on HeadVar__1: the disjunction has more than one disjunct with solutions
clauses(a, 1).
clauses(a, 4).

% A disjunction is where its first disjunct is, as in mmc.
disjuncts(X, Y) :- (
% CHECK-DAG: goal-errors.m:[[@LINE+2]]:5: because: the disjunction has more than one disjunct with solutions
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:5: because: the call to >(in, in) can fail
    X > 1, Y = 1
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:5: because: the call to <(in, in) can fail
  ; X < 0, Y = 2 ).

% A condition that can succeed more than once, and an else part that
% can fail.
if_then_else(X, Y) :-
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:8: because: the call to nd(in, out) can succeed more than once
  ( if nd(X, Z) then Y = Z
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:10: because: the call to sd(in, out) can fail
    else sd(X, Y) ).

% CHECK-DAG: goal-errors.m:[[@LINE+1]]:16: because: the negated goal can succeed
negation(X) :- not X > 0.

% CHECK-DAG: goal-errors.m:[[@LINE+2]]:29: because: the call to member(out, in) can fail
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:48: because: the call to >(in, in) can fail
some_scope(L) :- some [X] ( list.member(X, L), X > 3 ).

% CHECK-DAG: goal-errors.m:[[@LINE+3]]:23: because: the unification of Y and X can succeed
% CHECK-DAG: goal-errors.m:[[@LINE+2]]:34: because: the unification of V_2 and 0 can succeed
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:30: because: the call to >(in, in) can fail and can succeed
no_solutions(X, Y) :- Y = X, X > 0.

failing(_) :- fail.

% CHECK-DAG: goal-errors.m:[[@LINE+2]]:19: error: the lambda expression is declared det, but its determinism is inferred to be semidet
% CHECK-DAG: goal-errors.m:[[@LINE+1]]:42: because: the call to sd(in, out) can fail
lambda(P) :- P = (pred(Y::out) is det :- sd(1, Y)).

primary(X) = Y :- ( Y = X ; Y = X + 1 ).
