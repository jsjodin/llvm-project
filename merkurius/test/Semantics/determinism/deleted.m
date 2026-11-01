% Goals that mmc deletes or replaces before determinism analysis. A
% construction of a variable not used after it is deleted, but its
% variables still count as used, since mmc does not quantify the body
% again after it, so that a call before it is not pruned. The negation of
% true is fail, and the negation of a negation the goal inside. Of a
% committed choice call in a context that requires all solutions, the
% reasons are the goals after it that can fail, or those in the branches
% of one that cannot. After a goal that fails only by inference, the goals
% that can fail are still reasons, but not those that can succeed more
% than once.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: --implicit-check-not=because: %s < %t.err
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module deleted.
:- interface.
:- type t ---> t(int).

% ERR-DAG: deleted.m:[[@LINE+1]]:9: error: the determinism declaration of construction(out) is not satisfied: it is declared det, but inferred multi
:- pred construction(int::out) is det.
% ERR-DAG: deleted.m:[[@LINE+1]]:9: error: the determinism declaration of negated_construction(out) is not satisfied: it is declared det, but inferred failure
:- pred negated_construction(int::out) is det.
% ERR-DAG: deleted.m:[[@LINE+1]]:9: error: the determinism declaration of in_condition(out) is not satisfied: it is declared det, but inferred multi
:- pred in_condition(int::out) is det.
% ERR-DAG: deleted.m:[[@LINE+1]]:9: error: the determinism declaration of in_disjunct(out) is not satisfied: it is declared det, but inferred multi
:- pred in_disjunct(int::out) is det.
% ERR-DAG: deleted.m:[[@LINE+1]]:9: error: the determinism declaration of in_scope(out) is not satisfied: it is declared det, but inferred multi
:- pred in_scope(int::out) is det.
:- pred negated_true(int::out) is det.
:- pred double_negation(int::out) is det.
:- pred binds(int::out) is det.
% ERR-DAG: deleted.m:[[@LINE+1]]:9: error: the determinism declaration of de_morgan(out) is not satisfied: it is declared det, but inferred multi
:- pred de_morgan(int::out) is det.
% ERR-DAG: deleted.m:[[@LINE+1]]:9: error: the determinism declaration of cc_disjunction(out) is not satisfied: it is declared det, but inferred multi
:- pred cc_disjunction(int::out) is det.
% ERR-DAG: deleted.m:[[@LINE+1]]:9: error: the determinism declaration of cc_condition(out) is not satisfied: it is declared det, but inferred multi
:- pred cc_condition(int::out) is det.
% ERR-DAG: deleted.m:[[@LINE+1]]:9: error: the determinism declaration of after_failure(out) is not satisfied: it is declared det, but inferred failure
:- pred after_failure(int::out) is det.

:- implementation.
:- import_module int.

:- pred m(int::out) is multi.
m(1).
m(2).
:- pred cm(int::out) is cc_multi.
cm(1).
cm(2).
:- pred dt(int::out) is det.
dt(1).

% ERR-DAG: deleted.m:[[@LINE+1]]:20: because: the call to m(out) can succeed more than once
construction(Y) :- m(X), T = t(X), Y = 1.
% ERR-DAG: deleted.m:[[@LINE+1]]:34: because: the negated goal can succeed
negated_construction(Y) :- m(X), \+ ( T = t(X) ), Y = 1.
% ERR-DAG: deleted.m:[[@LINE+1]]:20: because: the call to m(out) can succeed more than once
in_condition(Y) :- m(X), ( if T = t(X) then Y = 1 else Y = 2 ).
% ERR-DAG: deleted.m:[[@LINE+2]]:19: because: the call to m(out) can succeed more than once
% ERR-DAG: deleted.m:[[@LINE+1]]:27: because: the disjunction has more than one disjunct with solutions
in_disjunct(Y) :- m(X), ( T = t(X), Y = 1 ; Y = 2 ).
% ERR-DAG: deleted.m:[[@LINE+1]]:16: because: the call to m(out) can succeed more than once
in_scope(Y) :- m(X), some [T] ( T = t(X) ), Y = 1.
negated_true(Y) :- ( Y = 1 ; \+ ( true, true ), Y = 2 ).
double_negation(Y) :- \+ \+ m(_), Y = 1.
% The goal inside may bind a variable used after it, as in mmc (see the
% design document's register).
binds(Y) :- \+ \+ X = 1, Y = X.
% ERR-DAG: deleted.m:[[@LINE+1]]:17: because: the disjunction has more than one disjunct with solutions
de_morgan(Y) :- \+ ( \+ Y = 1, \+ Y = 2 ).
% ERR-DAG: deleted.m:[[@LINE+4]]:22: because: the call to cm(out) can succeed more than once
% ERR-DAG: deleted.m:[[@LINE+3]]:31: because: the disjunction has more than one disjunct with solutions
% ERR-DAG: deleted.m:[[@LINE+2]]:22: error: the call to cm(out) with determinism cc_multi is in a context that requires all solutions
% ERR-DAG: deleted.m:[[@LINE+1]]:39: because: the call to >(in, in) can fail
cc_disjunction(Y) :- cm(X), ( Y = X ; X > 0, Y = 1 ).
% ERR-DAG: deleted.m:[[@LINE+2]]:20: because: the call to cm(out) can succeed more than once
% ERR-DAG: deleted.m:[[@LINE+1]]:20: error: the call to cm(out) with determinism cc_multi is in a context that requires all solutions
cc_condition(Y) :- cm(X), ( if X > 0 then Y = 1 else Y = 2 ).
% ERR-DAG: deleted.m:[[@LINE+2]]:21: because: the negated goal can succeed
% ERR-DAG: deleted.m:[[@LINE+1]]:37: because: the call to >(in, in) can fail
after_failure(Y) :- \+ dt(_), m(Y), Y > 1.
