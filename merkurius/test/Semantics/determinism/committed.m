% Committed choice (reference manual, "Committed choice nondeterminism"),
% as mmc checks it: the goals where only the first solution is wanted are
% the body of a committed choice procedure or lambda expression, a goal
% with no outputs, the last conjunct of a conjunction where it is wanted
% and those before conjuncts that cannot fail, the then and else parts of
% an if-then-else where it is wanted and its condition if the then part
% cannot fail, and the parts of disjunctions, switches and scopes where it
% is. There a goal with more than one solution is committed choice, unless
% one of its parts has more than one where all are wanted, and a committed
% choice call elsewhere is an error, with the goals after it that can fail.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err | FileCheck %s
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: --implicit-check-not=because: %s < %t.err
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module committed.
:- interface.
:- type t ---> a ; b ; c(int).

:- pred calls_multi(int::out) is cc_multi.
:- pred disjunction(int::out) is cc_multi.
:- pred if_then_else(int::in, int::out) is cc_multi.
:- pred condition(int::in, int::out) is cc_multi.
:- pred some_scope(int::out) is cc_multi.
:- pred negation(int::in) is semidet.
:- pred ignored(int::in) is det.
:- pred higher_order(pred(int)::in(pred(out) is cc_multi), int::out) is cc_multi.
:- pred lambda(pred(int)::out(pred(out) is cc_multi)) is det.
% ERR-DAG: committed.m:[[@LINE+1]]:9: warning: the determinism declaration of loose(out) could be tighter: it is declared cc_multi, but inferred det
:- pred loose(int::out) is cc_multi.
% ERR-DAG: committed.m:[[@LINE+1]]:9: error: the determinism declaration of all_solutions(out) is not satisfied: it is declared det, but inferred multi
:- pred all_solutions(int::out) is det.
:- pred multi(int::out) is multi.
% ERR-DAG: committed.m:[[@LINE+1]]:9: error: the determinism declaration of then_fails(in, out) is not satisfied: it is declared cc_multi, but inferred cc_nondet
:- pred then_fails(int::in, int::out) is cc_multi.
:- pred later(t::in, int::out) is cc_nondet.
:- pred outer(int::in, int::out) is cc_nondet.
:- pred condition_fails(int::in, int::out) is cc_nondet.
% ERR-DAG: committed.m:[[@LINE+1]]:9: error: the determinism declaration of all_higher_order(in((pred(out) is cc_multi)), out) is not satisfied: it is declared det, but inferred multi
:- pred all_higher_order(pred(int)::in(pred(out) is cc_multi), int::out) is det.
% ERR-DAG: committed.m:[[@LINE+1]]:9: error: the determinism declaration of function(out) is not satisfied: it is declared det, but inferred multi
:- pred function(int::out) is det.
:- pred lambda_all(pred(int)::out(pred(out) is det)) is det.

:- implementation.
:- import_module int.

:- pred m(int::out) is multi.
m(1).
m(2).
:- pred ccm(int::out) is cc_multi.
ccm(X) :- m(X).
:- pred ccn(int::in, int::out) is cc_nondet.
ccn(X, Y) :- X > 0, m(Y).
:- pred sd(int::in) is semidet.
sd(X) :- X > 0.
:- func ccf(int) = int.
% ERR-DAG: committed.m:[[@LINE+1]]:9: error: invalid determinism for ccf(in) = out: the primary mode of a function cannot be cc_multi
:- mode ccf(in) = out is cc_multi.
ccf(X) = Y :- m(Z), Y = X + Z.

% CHECK-LABEL: calls_multi(out) is cc_multi, inferred cc_multi
% CHECK: m(X) [multi]
calls_multi(X) :- m(X).

% CHECK-LABEL: disjunction(out) is cc_multi, inferred cc_multi
% CHECK: disjunction [cc_multi]
disjunction(X) :- ( ccm(X) ; X = 3 ).

% CHECK-LABEL: if_then_else(in, out) is cc_multi, inferred cc_multi
% CHECK: if-then-else [cc_multi]
if_then_else(N, X) :- ( if N > 0 then ccm(X) else X = 0 ).

% CHECK-LABEL: condition(in, out) is cc_multi, inferred cc_multi
% CHECK: ccn(N, Y) [cc_nondet]
condition(N, X) :- ( if ccn(N, Y) then X = Y else X = 0 ).

% CHECK-LABEL: some_scope(out) is cc_multi, inferred cc_multi
some_scope(X) :- some [Y] ( ccm(Y), X = Y ).

% CHECK-LABEL: negation(in) is semidet, inferred semidet
negation(N) :- not ccn(N, _).

% Each conjunct has no outputs.
% CHECK-LABEL: ignored(in) is det, inferred det
ignored(N) :- ( if ccm(_), N > 0 then true else true ).

% CHECK-LABEL: higher_order(in((pred(out) is cc_multi)), out) is cc_multi, inferred cc_multi
higher_order(P, X) :- P(X).

% CHECK-LABEL: lambda(out((pred(out) is cc_multi))) is det, inferred det
% CHECK: m(X) [multi]
lambda(P) :- P = (pred(X::out) is cc_multi :- m(X)).

loose(1).

% A disjunction with more than one solution where all are wanted is not
% committed choice.
% CHECK-LABEL: multi(out) is multi, inferred multi
% CHECK: disjunction [multi]
multi(X) :- ( X = 1 ; X = 2 ).

% ERR-DAG: committed.m:[[@LINE+2]]:21: because: the call to ccm(out) can succeed more than once
% ERR-DAG: committed.m:[[@LINE+1]]:21: error: the call to ccm(out) with determinism cc_multi is in a context that requires all solutions
all_solutions(X) :- ccm(X).

% ERR-DAG: committed.m:[[@LINE+2]]:21: error: the call to ccm(out) with determinism cc_multi is in a context that requires all solutions
% ERR-DAG: committed.m:[[@LINE+1]]:29: because: the call to >(in, in) can fail
then_fails(N, X) :- ccm(X), X > N.

% The goals after it that can fail, in parts of them, and around it.
% ERR-DAG: committed.m:[[@LINE+2]]:16: error: the call to ccm(out) with determinism cc_multi is in a context that requires all solutions
% ERR-DAG: committed.m:[[@LINE+1]]:26: because: the switch on T does not cover c/1
later(T, X) :- ccm(X), ( T = a ; T = b ).
% ERR-DAG: committed.m:[[@LINE+2]]:27: error: the call to ccm(out) with determinism cc_multi is in a context that requires all solutions
% ERR-DAG: committed.m:[[@LINE+1]]:44: because: the call to sd(in) can fail
outer(N, X) :- some [Z] ( ccm(Z), X = Z ), sd(N).

% A condition whose then part can fail, which makes the if-then-else not
% committed choice.
% CHECK-LABEL: condition_fails(in, out) is cc_nondet, inferred cc_nondet
% CHECK: if-then-else [nondet]
% ERR-DAG: committed.m:[[@LINE+2]]:31: error: the call to ccm(out) with determinism cc_multi is in a context that requires all solutions
% ERR-DAG: committed.m:[[@LINE+1]]:43: because: the call to sd(in) can fail
condition_fails(N, X) :- ( if ccm(Y) then sd(Y), X = Y else X = N ).

% ERR-DAG: committed.m:[[@LINE+2]]:27: because: the higher-order call can succeed more than once
% ERR-DAG: committed.m:[[@LINE+1]]:27: error: the higher-order call with determinism cc_multi is in a context that requires all solutions
all_higher_order(P, X) :- P(X).

% ERR-DAG: committed.m:[[@LINE+2]]:20: because: the call to ccf(in) = out can succeed more than once
% ERR-DAG: committed.m:[[@LINE+1]]:20: error: the call to ccf(in) = out with determinism cc_multi is in a context that requires all solutions
function(X) :- X = ccf(1).

% ERR-DAG: committed.m:[[@LINE+3]]:23: error: the lambda expression is declared det, but its determinism is inferred to be multi
% ERR-DAG: committed.m:[[@LINE+2]]:46: because: the call to ccm(out) can succeed more than once
% ERR-DAG: committed.m:[[@LINE+1]]:46: error: the call to ccm(out) with determinism cc_multi is in a context that requires all solutions
lambda_all(P) :- P = (pred(X::out) is det :- ccm(X)).
