% Unique values needed on backtracking in lambda expressions, closures,
% higher-order calls and scopes, as mmc's second phase of unique mode
% analysis finds them. A lambda expression's body is checked where it is,
% as a procedure of its own; a closure of a predicate that can succeed more
% than once leaves its outputs mostly_unique; a higher-order call's
% arguments must match its term's modes. A promise_equivalent_solutions or
% arbitrary scope, and a goal with no outputs, commits, so that what
% follows it is not run again on backtracking into it, but a
% promise_equivalent_solution_sets scope does not. In an if-then-else,
% what is used after it is not taken as needed on backtracking into it,
% as in mmc (see the design document's register).
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: %s < %t.err
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module scopes.
:- interface.
:- type t ---> t(int).

:- pred in_lambda(t::di, t::uo) is det.
:- pred lambda_disjunction(t::di, t::uo) is det.
:- pred cc_lambda(t::di, t::uo) is cc_multi.
:- pred lambda_condition(t::di, t::uo) is det.
:- pred lambda_final(t::di, t::uo) is det.
:- pred nested_lambda(t::di, t::uo) is det.
:- pred multi_closure(t::di, t::uo) is det.
:- pred det_closure(t::di, t::uo) is det.
:- pred closure_argument(t::di, t::uo) is multi.
:- pred higher_order_output(t::di, t::uo) is multi.
:- pred promised_before(t::di, t::uo) is semidet.
:- pred arbitrary_before(t::di, t::uo) is semidet.
:- pred solution_sets(t::di, t::uo) is multi.
:- pred condition_then(t::di, t::uo) is multi.
:- pred after_condition(t::di, t::uo) is multi.
:- pred after_then(t::di, t::uo) is multi.
:- pred after_disjunction(t::di, t::uo) is multi.
:- pred no_outputs(t::di, t::uo) is det.
:- pred no_outputs_erroneous(t::di, t::uo) is det.

:- implementation.
:- import_module int.

:- pred updu(int::in, t::di, t::uo) is det.
updu(Y, t(X), t(X + Y)).
:- pred m(int::out) is multi.
m(1).
m(2).
:- pred sd(int::in) is semidet.
sd(X) :- X > 0.
:- pred mmu(t::mdi, t::muo) is multi.
mmu(T, T).
mmu(t(X), t(X + 1)).
% Its own final inst is needed on backtracking.
:- pred mu(t::di, t::uo) is multi.
mu(T, T).
% ERR-DAG: scopes.m:[[@LINE+1]]:10: error: in the clause for mu(di, uo): mode error: argument 2, HeadVar__2, did not get sufficiently instantiated: its final inst is mostly_unique, but unique is expected
mu(t(X), t(X + 1)).
:- pred loop(t::mdi, t::muo).
loop(T0, T) :- loop(T0, T).

% ERR-DAG: scopes.m:[[@LINE+1]]:63: error: in the clause for in_lambda(di, uo): mode error in argument 2 of the call to pred scopes.updu/3: A has inst mostly_unique, but unique is expected
in_lambda(T0, T) :- P = (pred(A::di, B::uo) is multi :- m(X), updu(X, A, B)), promise_equivalent_solutions [T] P(T0, T).
% ERR-DAG: scopes.m:[[@LINE+1]]:68: error: in the clause for lambda_disjunction(di, uo): mode error in argument 2 of the call to pred scopes.updu/3: A has inst mostly_unique, but unique is expected
lambda_disjunction(T0, T) :- P = (pred(A::di, B::uo) is multi :- ( updu(1, A, B) ; updu(2, A, B) )), promise_equivalent_solutions [T] P(T0, T).
cc_lambda(T0, T) :- P = (pred(A::di, B::uo) is cc_multi :- m(X), updu(X, A, B)), P(T0, T).
% ERR-DAG: scopes.m:[[@LINE+1]]:67: error: in the clause for lambda_condition(di, uo): mode error in argument 2 of the call to pred scopes.updu/3: A has inst mostly_unique, but unique is expected
lambda_condition(T0, T) :- P = (pred(A::di, B::uo) is det :- ( if updu(1, A, B1), 1 > 2 then B = B1 else B = A )), P(T0, T).
% ERR-DAG: scopes.m:[[@LINE+1]]:92: error: in the clause for lambda_final(di, uo): mode error: argument 2, LambdaHeadVar__2, did not get sufficiently instantiated: its final inst is mostly_unique, but unique is expected
lambda_final(T0, T) :- P = (pred(A::di, B::uo) is multi :- m(X), B0 = A, ( X = 1, B = B0 ; B = B0 )), promise_equivalent_solutions [T] P(T0, T).
% ERR-DAG: scopes.m:[[@LINE+1]]:101: error: in the clause for nested_lambda(di, uo): mode error in argument 2 of the call to pred scopes.updu/3: C has inst mostly_unique, but unique is expected
nested_lambda(T0, T) :- P = (pred(A::di, B::uo) is det :- Q = (pred(C::di, D::uo) is multi :- m(X), updu(X, C, D)), promise_equivalent_solutions [B] Q(A, B)), P(T0, T).
% ERR-DAG: scopes.m:[[@LINE+1]]:80: error: in the clause for multi_closure(di, uo): mode error: argument 2, T, did not get sufficiently instantiated: its final inst is mostly_unique, but unique is expected
multi_closure(T0, T) :- P = mmu, promise_equivalent_solutions [T] ( P(T0, T1), T = T1 ).
det_closure(T0, T) :- P = updu(1), promise_equivalent_solutions [T] ( m(X), P(T0, T1), updu(X, T1, T) ).
% ERR-DAG: scopes.m:[[@LINE+1]]:34: error: in the clause for closure_argument(di, uo): mode error in argument 2 of the call to pred scopes.updu/3: T0 has inst mostly_unique, but unique is expected
closure_argument(T0, T) :- m(X), updu(X, T0, T1), P = updu(1), P(T1, T).
% ERR-DAG: scopes.m:[[@LINE+1]]:86: error: in the clause for higher_order_output(di, uo): mode error in argument 2 of the call to pred scopes.updu/3: T1 has inst mostly_unique, but unique is expected
higher_order_output(T0, T) :- P = (pred(A::di, B::uo) is multi :- B = A), P(T0, T1), updu(1, T1, T).
promised_before(T0, T) :- promise_equivalent_solutions [X] ( m(X), X > 1 ), updu(X, T0, T).
arbitrary_before(T0, T) :- promise_equivalent_solution_sets [T] ( arbitrary [X] ( m(X), X > 1 ), updu(X, T0, T) ).
% ERR-DAG: scopes.m:[[@LINE+1]]:98: error: in the clause for solution_sets(di, uo): mode error in argument 2 of the call to pred scopes.updu/3: T1 has inst mostly_unique, but unique is expected
solution_sets(T0, T) :- promise_equivalent_solution_sets [T] ( m(X), arbitrary [T1] ( T1 = T0 ), updu(X, T1, T) ).
% ERR-DAG: scopes.m:[[@LINE+1]]:48: error: in the clause for condition_then(di, uo): mode error in argument 2 of the call to pred scopes.updu/3: T0 has inst mostly_unique, but unique is expected
condition_then(T0, T) :- ( if m(X), X > 1 then updu(X, T0, T) else T = t(0) ).
after_condition(T0, T) :- ( if m(X), X > 1 then Y = X else Y = 0 ), updu(Y, T0, T).
after_then(T0, T) :- ( if 1 > 2 then m(Y) else Y = 2 ), updu(Y, T0, T).
% ERR-DAG: scopes.m:[[@LINE+1]]:47: error: in the clause for after_disjunction(di, uo): mode error in argument 2 of the call to pred scopes.updu/3: T0 has inst mostly_unique, but unique is expected
after_disjunction(T0, T) :- ( m(X) ; X = 3 ), updu(X, T0, T).
no_outputs(T0, T) :- ( if m(X), sd(X) then C = T0 else updu(3, T0, C) ), T = C.
no_outputs_erroneous(T0, T) :- ( if mu(T0, T1), updu(1, T1, T2) then loop(T2, T3), T4 = T3 else T4 = t(1) ), T = t(2).
