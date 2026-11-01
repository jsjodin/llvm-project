% Unique values needed on backtracking, as mmc's second phase of unique
% mode analysis finds them: after a goal that can succeed more than once,
% in a disjunction, in the condition of an if-then-else, and in a negated
% goal, a unique value bound before is mostly_unique, so that a call that
% would update it, or a head variable that must end unique, is an error.
% Where only the first solution is wanted, a goal is backtracked into only
% if a goal after it can fail. Goals whose end mode analysis does not
% reach are left out; goals after one that fails only by inference are
% not.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: %s < %t.err
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module backtracking.
:- interface.
:- type t ---> t(int).
:- type e ---> e1 ; e2.

:- pred after_multi(t::di, t::muo) is multi.
:- pred assigned_after(t::di, t::muo) is multi.
:- pred no_outputs(t::di, t::uo) is det.
:- pred callee_output(t::uo) is multi.
:- pred cc_call(t::di, t::uo) is cc_multi.
:- pred semidet_disj(t::di, t::uo) is nondet.
:- pred cc_disj(t::di, t::uo) is cc_multi.
:- pred cc_disj_fails(t::di, t::uo) is cc_multi.
:- pred no_outputs_fails(t::di, t::uo) is det.
:- pred cc_fails_after(t::di, t::uo) is cc_nondet.
:- pred promised_fails(t::di, t::uo) is det.
:- pred promised_last(t::di, t::uo) is det.
:- pred promised_nested(t::mdi, t::muo) is cc_nondet.
:- pred condition(t::di, t::muo) is det.
:- pred condition_alias(t::di, t::muo) is det.
:- pred then_part(t::di, t::muo) is det.
:- pred negated(t::di, t::muo) is semidet.
:- pred negated_in_condition(t::di, t::muo) is det.
:- pred unreached_disjunct(t::di, t::uo) is semidet.
:- pred inferred_failure(t::di, t::muo) is multi.
:- pred final(t::di, int::out, t::uo) is multi.
:- pred nested_bound(t::muo) is multi.
:- pred constant(e::muo) is multi.

:- implementation.
:- import_module int.

:- pred updu(int::in, t::di, t::uo) is det.
updu(Y, t(X), t(X + Y)).
:- pred m(int::out) is multi.
m(1).
m(2).
:- pred mt(t::uo) is multi.
mt(t(1)).
mt(t(2)).
:- pred cm(int::out) is cc_multi.
cm(1).
:- pred dt(int::out) is det.
dt(1).

% ERR-DAG: backtracking.m:[[@LINE+1]]:38: error: in the clause for after_multi(di, muo): mode error in argument 2 of the call to pred backtracking.updu/3: T1 has inst mostly_unique, but unique is expected
after_multi(T0, T) :- T1 = T0, m(X), updu(X, T1, T).
% ERR-DAG: backtracking.m:[[@LINE+1]]:41: error: in the clause for assigned_after(di, muo): mode error in argument 2 of the call to pred backtracking.updu/3: T1 has inst mostly_unique, but unique is expected
assigned_after(T0, T) :- m(X), T1 = T0, updu(X, T1, T).
% A call whose outputs are not used has at most one solution.
no_outputs(T0, T) :- m(_), updu(1, T0, T).
% ERR-DAG: backtracking.m:[[@LINE+1]]:29: error: in the clause for callee_output(uo): mode error in argument 2 of the call to pred backtracking.updu/3: T0 has inst mostly_unique, but unique is expected
callee_output(T) :- mt(T0), updu(1, T0, T).
% A committed choice call is never backtracked into.
cc_call(T0, T) :- cm(X), updu(X, T0, T).

% Any disjunction that can succeed more than once, even in its last
% disjunct.
% ERR-DAG: backtracking.m:[[@LINE+1]]:51: error: in the clause for semidet_disj(di, uo): mode error in argument 2 of the call to pred backtracking.updu/3: T0 has inst mostly_unique, but unique is expected
semidet_disj(T0, T) :- ( 1 > 2, T = t(1) ; 2 > 1, updu(1, T0, T) ).
% Where only the first solution is wanted, as in one whose outputs are not
% used, a disjunct that cannot fail is the last one run, and a disjunct
% that can fail makes the values it uses mostly_unique in it only.
cc_disj(T0, T) :- ( T = t(1) ; T1 = T0, T = T1 ; updu(1, T0, T) ).
cc_disj_fails(T0, T) :- ( 1 > 2, T = t(1) ; T = t(2) ; updu(1, T0, T) ).
% ERR-DAG: backtracking.m:[[@LINE+1]]:54: error: in the clause for no_outputs_fails(di, uo): mode error in argument 2 of the call to pred backtracking.updu/3: T0 has inst mostly_unique, but unique is expected
no_outputs_fails(T0, T) :- ( updu(1, T0, _) ; true ; updu(1, T0, _), 1 > 2 ), T = t(3).
% ERR-DAG: backtracking.m:[[@LINE+1]]:39: error: in the clause for cc_fails_after(di, uo): mode error in argument 2 of the call to pred backtracking.updu/3: T0 has inst mostly_unique, but unique is expected
cc_fails_after(T0, T) :- m(X), X > 1, updu(X, T0, T).
% ERR-DAG: backtracking.m:[[@LINE+1]]:77: error: in the clause for promised_fails(di, uo): mode error: argument 2, T, did not get sufficiently instantiated: its final inst is mostly_unique, but unique is expected
promised_fails(T0, T) :- promise_equivalent_solutions [T] ( 1 > 2, T = T0 ; updu(1, T0, T) ).
% ERR-DAG: backtracking.m:[[@LINE+1]]:91: error: in the clause for promised_last(di, uo): mode error in argument 2 of the call to pred backtracking.updu/3: T1 has inst mostly_unique, but unique is expected
promised_last(T0, T) :- promise_equivalent_solutions [T1] ( T1 = t(1) ; 1 > 2, T1 = T0 ), updu(1, T1, T).
% ERR-DAG: backtracking.m:[[@LINE+2]]:36: error: in the clause for promised_nested(mdi, muo): mode error in argument 2 of the call to pred backtracking.updu/3: T9 has inst mostly_unique, but unique is expected
promised_nested(T0, T) :-
    ( m(_), T5 = t(0) ; ( T9 = T0, updu(3, T9, T8) ; 2 > 1, T8 = T0 ; T8 = t(1) ), T5 = T8 ),
    updu(1, T5, T).

% The condition's values that the else part uses, but not what is made
% from them in the condition, though that may clobber them (see the
% design document's register).
% ERR-DAG: backtracking.m:[[@LINE+1]]:26: error: in the clause for condition(di, muo): mode error in argument 2 of the call to pred backtracking.updu/3: T0 has inst mostly_unique, but unique is expected
condition(T0, T) :- ( if updu(1, T0, T5), 1 > 2 then T = T5 else T = T0 ).
condition_alias(T0, T) :- ( if T4 = T0, updu(1, T4, T5), 1 > 2 then T = T5 else T = T0 ).
then_part(T0, T) :- ( if T4 = T0, 1 > 2 then updu(1, T4, T) else T = T0 ).

% The values used after a negation, inside it.
% ERR-DAG: backtracking.m:[[@LINE+1]]:24: error: in the clause for negated(di, muo): mode error in argument 2 of the call to pred backtracking.updu/3: T0 has inst mostly_unique, but unique is expected
negated(T0, T) :- \+ ( updu(1, T0, T6), updu(2, T6, _) ), updu(1, T0, T).
% ERR-DAG: backtracking.m:[[@LINE+1]]:51: error: in the clause for negated_in_condition(di, muo): mode error in argument 2 of the call to pred backtracking.updu/3: T4 has inst mostly_unique, but unique is expected
negated_in_condition(T0, T) :- ( if T4 = T0, \+ ( updu(1, T4, T6), updu(2, T6, _) ) then T = T4 else T = T0 ).

% mmc takes the negation of a negation as the goal inside, after which
% what is used after it is live.
:- pred double_negation(t::di, t::muo) is semidet.
% ERR-DAG: backtracking.m:[[@LINE+1]]:37: error: in the clause for double_negation(di, muo): mode error in argument 2 of the call to pred backtracking.updu/3: T0 has inst ground, but unique is expected
double_negation(T0, T) :- \+ ( \+ ( updu(1, T0, T1), T1 = t(2) ) ), T = T0.

% mmc takes the negation of true as fail, and leaves out the disjunct.
unreached_disjunct(T0, T) :- ( 1 > 2, T1 = T0 ; \+ true, T1 = T0 ), updu(2, T1, T).
% But not goals after one that fails only by inference.
% ERR-DAG: backtracking.m:[[@LINE+1]]:62: error: in the clause for inferred_failure(di, muo): mode error in argument 2 of the call to pred backtracking.updu/3: T1 has inst mostly_unique, but unique is expected
inferred_failure(T0, T) :- ( if \+ dt(_) then m(X), T1 = T0, updu(X, T1, T) else T = T0 ).

% ERR-DAG: backtracking.m:[[@LINE+1]]:26: error: in the clause for final(di, out, uo): mode error: argument 3, T, did not get sufficiently instantiated: its final inst is mostly_unique, but unique is expected
final(T0, X, T) :- m(X), T = T0.
% A bound inst made mostly_unique with a bound argument does not match
% ground(mostly_unique) in mmc, though a constant does (see the design
% document's register).
% ERR-DAG: backtracking.m:[[@LINE+1]]:42: error: in the clause for nested_bound(muo): mode error: argument 1, T, had the wrong inst: its final inst is mostly_unique(t(mostly_unique(3))), but mostly_unique is expected
nested_bound(T) :- T6 = t(3), ( T = T6 ; T = T6 ).
constant(T) :- T6 = e1, ( T = T6 ; T = T6 ).
