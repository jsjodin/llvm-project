% Malformed goals and clauses: a DCG lambda expression without its two DCG
% modes, trace and atomic parameters that are not ones, an atomic goal with
% a parameter twice or without its inner state, a catch clause that is not
% 'Pattern -> Goal', a switch requirement on two variables, an event that
% is not a call, a DCG input match with a tail, a DCG goal that is a number,
% and a clause head with modes on some arguments. mmc reports an error on
% each of these lines too: after parsing for the switch requirement, the
% event, the clause head and the atomic goals, and as a type error for the
% number.
%
% RUN: not merkc -fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module goal_errors.
:- implementation.
:- import_module int, list.
:- pred dl(int::out) is det.
dl(X) :- P = (pred(in) is det --> []), X = 1.
% CHECK: goal-errors.m:[[@LINE-1]]:15: error: a DCG lambda expression's head must end with the modes of its two DCG arguments, as in 'pred(in, out) is det --> G'
:- pred tc(int::out) is det.
tc(X) :- trace [compile_time(foo(x))] true, X = 1.
% CHECK: goal-errors.m:[[@LINE-1]]:30: error: expected a compile_time condition: flag("Name"), tracelevel(Level) or grade(Test), with and, or and not, but found 'foo(x)'
:- pred tr(int::out) is det.
tr(X) :- trace [run_time(bar)] true, X = 1.
% CHECK: goal-errors.m:[[@LINE-1]]:26: error: expected a run_time condition: env("Name"), with and, or and not, but found 'bar'
:- pred tp(int::out) is det.
tp(X) :- trace [frobnicate] true, X = 1.
% CHECK: goal-errors.m:[[@LINE-1]]:17: error: expected a trace parameter: compile_time(...), run_time(...), io(!IO) or state(Mutable, !S), but found 'frobnicate'
:- pred cc(int::out) is cc_multi.
cc(X) :- try [] X = 1 then true catch foo.
% CHECK: goal-errors.m:[[@LINE-1]]:39: error: expected 'Pattern -> Goal', but found 'foo'
:- pred sw(int::in, int::in) is semidet.
sw(X, Y) :- require_complete_switch [X, Y] ( X = 1 ; Y = 2 ).
% CHECK: goal-errors.m:[[@LINE-1]]:37: error: expected a list of one variable, but found '[X, Y]'
:- pred ev(int::out) is det.
ev(X) :- event 42, X = 1.
% CHECK: goal-errors.m:[[@LINE-1]]:16: error: expected a predicate call, but found '42'
:- pred dt(list(int)::in, list(int)::out) is semidet.
dt --> [1 | T].
% CHECK: goal-errors.m:[[@LINE-1]]:8: error: expected a list without a tail, but found '[1 | T]'
:- pred ds(list(int)::in, list(int)::out) is semidet.
ds --> 42.
% CHECK: goal-errors.m:[[@LINE-1]]:8: error: expected a DCG goal, but found '42'
:- pred mm(int::in, int::out) is det.
mm(X::in, Y) :- Y = X.
% CHECK: goal-errors.m:[[@LINE-1]]:1: error: either every argument or none must have a mode
:- pred at(int::out) is det.
at(X) :- atomic [frob(Y)] ( X = 1 ).
% CHECK: goal-errors.m:[[@LINE-1]]:18: error: expected an atomic parameter: outer(!S), inner(!S), outer(S0, S), inner(S0, S) or vars([Vars]), but found 'frob(Y)'
:- pred at2(int::out) is det.
at2(X) :- atomic [outer(A, B), outer(C, D), inner(E, F)] ( X = 1 ).
% CHECK: goal-errors.m:[[@LINE-1]]:18: error: the atomic parameter 'outer' is given more than once
:- pred at3(int::out) is det.
at3(X) :- atomic [outer(A, B)] ( X = 1 ).
% CHECK: goal-errors.m:[[@LINE-1]]:18: error: an atomic goal needs the parameters outer(...) and inner(...), but inner(...) is missing
