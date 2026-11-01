% Determinism requirements (reference manual, "Goals"), as mmc checks
% them: the goal of require_<det> must have that determinism exactly;
% that of require_complete_switch [X] must be a switch on X with a case for
% each functor X can be bound to; and that of require_switch_arms_<det>
% [X] a switch on X whose arms make the promises <det> makes. A goal with
% more than one solution is committed choice where only its first is
% wanted.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err | FileCheck %s
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: --implicit-check-not=because: %s < %t.err
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module requires.
:- interface.
:- type t ---> a ; b ; c(int).

:- pred det_ok(int::in, int::out) is det.
:- pred semidet_goal(int::in, int::out) is semidet.
:- pred multi_goal(int::out) is multi.
:- pred tighter(int::in, int::out) is semidet.
:- pred no_failure(int::in) is semidet.
:- pred cc_ok(int::out) is cc_multi.
% ERR-DAG: requires.m:[[@LINE+1]]:9: error: the determinism declaration of cc_all(out) is not satisfied: it is declared det, but inferred multi
:- pred cc_all(int::out) is det.
:- pred complete(t::in, int::out) is det.
:- pred incomplete(t::in, int::out) is semidet.
:- pred bound_inst(t::in(bound(a ; b)), int::out) is det.
:- pred int_switch(int::in, int::out) is semidet.
:- pred conjunction(t::in, int::out) is det.
:- pred alias(t::in, int::out) is det.
:- pred one_disjunct(t::in, int::out) is semidet.
:- pred other_var(t::in, t::in, int::out) is semidet.
:- pred arms(t::in, int::out) is semidet.
:- pred merged(t::in, int::out) is semidet.
% ERR-DAG: requires.m:[[@LINE+1]]:9: error: the determinism declaration of arms_semidet(in, out) is not satisfied: it is declared det, but inferred semidet
:- pred arms_semidet(t::in, int::out) is det.
:- pred arms_cc(t::in, int::out) is multi.
:- pred arms_cc_ok(t::in, int::out) is cc_multi.
% ERR-DAG: requires.m:[[@LINE+1]]:9: error: the determinism declaration of arms_erroneous(in, out) is not satisfied: it is declared det, but inferred semidet
:- pred arms_erroneous(t::in, int::out) is det.
:- pred arms_not_switch(t::in, t::in, int::out) is det.
:- pred arms_other_switch(t::in, t::in, int::out) is semidet.
:- pred in_cases(t::in, int::out) is semidet.
:- pred in_lambda(t::in, int::out) is det.
:- pred in_clause(t::in, t::in, int::out) is semidet.

:- implementation.
:- import_module int.

:- pred m(int::out) is multi.
m(1).
m(2).
:- pred cm(int::out) is cc_multi.
cm(1).
cm(2).

% CHECK-LABEL: pred requires.det_ok/2 mode 0: det_ok(in, out) is det, inferred det
% CHECK-NEXT:  require_det [det]
det_ok(X, Y) :- require_det ( Y = X + 1 ).

% ERR-DAG: requires.m:[[@LINE+2]]:23: error: the goal in this scope is required to be det, but its determinism is semidet
% ERR-DAG: requires.m:[[@LINE+1]]:37: because: the call to >(in, in) can fail
semidet_goal(X, Y) :- require_det ( X > 0, Y = X ).

% ERR-DAG: requires.m:[[@LINE+2]]:18: error: the goal in this scope is required to be det, but its determinism is multi
% ERR-DAG: requires.m:[[@LINE+1]]:30: because: the call to m(out) can succeed more than once
multi_goal(Y) :- require_det m(Y).

% A determinism tighter than required is an error too, with no reasons.
% ERR-DAG: requires.m:[[@LINE+1]]:18: error: the goal in this scope is required to be semidet, but its determinism is det
tighter(X, Y) :- require_semidet ( Y = X ).

% ERR-DAG: requires.m:[[@LINE+3]]:18: error: the goal in this scope is required to be failure, but its determinism is semidet
% ERR-DAG: requires.m:[[@LINE+2]]:40: because: the unification of V_1 and 0 can succeed
% ERR-DAG: requires.m:[[@LINE+1]]:36: because: the call to >(in, in) can succeed
no_failure(X) :- require_failure ( X > 0 ).

% m(Y) is committed choice here, as only its first solution is wanted.
cc_ok(Y) :- require_cc_multi m(Y).

% And cm(Y) is not.
% ERR-DAG: requires.m:[[@LINE+4]]:26: error: the call to cm(out) with determinism cc_multi is in a context that requires all solutions
% ERR-DAG: requires.m:[[@LINE+3]]:14: error: the goal in this scope is required to be det, but its determinism is multi
% ERR-DAG: requires.m:[[@LINE+2]]:26: because: the call to cm(out) can succeed more than once
% ERR-DAG: requires.m:[[@LINE+1]]:26: because: the call to cm(out) can succeed more than once
cc_all(Y) :- require_det cm(Y).

% CHECK-LABEL: pred requires.complete/2 mode 0: complete(in, out) is det, inferred det
% CHECK-NEXT:  require_complete_switch [X] [det]
% CHECK-NEXT:    switch on X, cannot fail [det]
complete(X, Y) :- require_complete_switch [X] ( X = a, Y = 1 ; X = b, Y = 2 ;
  X = c(_), Y = 3 ).

% ERR-DAG: requires.m:[[@LINE+1]]:21: error: the switch on X is required to be complete, but it does not cover c/1
incomplete(X, Y) :- require_complete_switch [X] ( X = a, Y = 1 ; X = b, Y = 2 ).

% Complete for the functors X can be bound to.
bound_inst(X, Y) :- require_complete_switch [X] ( X = a, Y = 1 ; X = b, Y = 2 ).

% ERR-DAG: requires.m:[[@LINE+1]]:21: error: the switch on X is required to be complete, but it is not
int_switch(X, Y) :- require_complete_switch [X] ( X = 1, Y = 1 ; X = 2, Y = 2 ).

% ERR-DAG: requires.m:[[@LINE+1]]:22: error: the goal inside the require_complete_switch [X] scope is not a switch on X
conjunction(X, Y) :- require_complete_switch [X] ( Z = 1,
  ( X = a, Y = Z ; X = b, Y = 2 ; X = c(_), Y = 3 ) ).

% A switch on W, not X.
% ERR-DAG: requires.m:[[@LINE+1]]:16: error: the goal inside the require_complete_switch [X] scope is not a switch on X
alias(X, Y) :- require_complete_switch [X] ( W = X,
  ( W = a, Y = 1 ; W = b, Y = 2 ; W = c(_), Y = 3 ) ).

% ERR-DAG: requires.m:[[@LINE+1]]:23: error: the goal inside the require_complete_switch [X] scope is not a switch on X
one_disjunct(X, Y) :- require_complete_switch [X] ( X = a, Y = 1 ).

% ERR-DAG: requires.m:[[@LINE+1]]:23: error: the goal inside the require_complete_switch [X] scope is not a switch on X
other_var(X, Z, Y) :- require_complete_switch [X] ( Z = a, Y = 1 ; Z = b, Y = 2 ).

% ERR-DAG: requires.m:[[@LINE+5]]:7: error: the arms of the switch on X are required to have a determinism acceptable in a det context, but the arm for b/0 is semidet
% ERR-DAG: requires.m:[[@LINE+5]]:7: error: the arms of the switch on X are required to have a determinism acceptable in a det context, but the arm for c/1 is semidet
% ERR-DAG: requires.m:[[@LINE+3]]:17: because: the call to >(in, in) can fail
% ERR-DAG: requires.m:[[@LINE+3]]:13: because: the call to >(in, in) can fail
arms(X, Y) :- require_switch_arms_det [X] ( X = a, Y = 1 ;
  X = b, Y = 3, Y > 2 ;
  X = c(Y), Y > 0 ).

% ERR-DAG: requires.m:[[@LINE+2]]:49: error: the arms of the switch on X are required to have a determinism acceptable in a det context, but the arm for b/0 and c/1 is semidet
% ERR-DAG: requires.m:[[@LINE+2]]:10: because: the call to >(in, in) can fail
merged(X, Y) :- require_switch_arms_det [X] ( ( X = b ; X = c(_) ),
  Y = 2, Y > 1 ; X = a, Y = 1 ).

% A det arm is acceptable in a semidet context; the switch need not be
% complete.
% ERR-DAG: requires.m:[[@LINE+1]]:57: because: the switch on X does not cover c/1
arms_semidet(X, Y) :- require_switch_arms_semidet [X] ( X = a, Y = 1 ;
  X = b, Y = 2 ).

% A multi arm is not acceptable in a cc_multi context unless only its
% first solution is wanted.
% ERR-DAG: requires.m:[[@LINE+1]]:57: error: the arms of the switch on X are required to have a determinism acceptable in a cc_multi context, but the arm for a/0 is multi
arms_cc(X, Y) :- require_switch_arms_cc_multi [X] ( X = a, m(Y) ; X = b, Y = 2 ;
  X = c(_), Y = 3 ).

arms_cc_ok(X, Y) :- require_switch_arms_cc_multi [X] ( X = a, m(Y) ;
  X = b, Y = 2 ; X = c(_), Y = 3 ).

% ERR-DAG: requires.m:[[@LINE+8]]:65: error: the arms of the switch on X are required to have a determinism acceptable in a erroneous context, but the arm for a/0 is det
% ERR-DAG: requires.m:[[@LINE+8]]:7: error: the arms of the switch on X are required to have a determinism acceptable in a erroneous context, but the arm for b/0 is failure
% ERR-DAG: requires.m:[[@LINE+7]]:21: error: the arms of the switch on X are required to have a determinism acceptable in a erroneous context, but the arm for c/1 is det
% ERR-DAG: requires.m:[[@LINE+5]]:65: because: the unification of X and a/0 can succeed
% ERR-DAG: requires.m:[[@LINE+4]]:72: because: the unification of Y and 1 can succeed
% ERR-DAG: requires.m:[[@LINE+4]]:7: because: the unification of X and b/0 can succeed
% ERR-DAG: requires.m:[[@LINE+3]]:21: because: the unification of X and c/1 can succeed
% ERR-DAG: requires.m:[[@LINE+2]]:31: because: the unification of Y and 3 can succeed
arms_erroneous(X, Y) :- require_switch_arms_erroneous [X] ( X = a, Y = 1 ;
  X = b, fail ; X = c(_), Y = 3 ).

% ERR-DAG: requires.m:[[@LINE+1]]:29: error: the goal inside the require_switch_arms_det [Z] scope is not a switch on Z
arms_not_switch(_, Z, Y) :- require_switch_arms_det [Z] ( Y = 1 ).

% When the goal is a switch on another variable, mmc checks its arms as if
% it were on Z, though the manual says it must report that it is not (see
% the register).
% ERR-DAG: requires.m:[[@LINE+3]]:7: error: the arms of the switch on X are required to have a determinism acceptable in a det context, but the arm for b/0 is semidet
% ERR-DAG: requires.m:[[@LINE+2]]:10: because: the call to >(in, in) can fail
arms_other_switch(X, Z, Y) :- require_switch_arms_det [Z] ( X = a, Y = 1 ;
  X = b, 2 > 1, Y = 2 ; X = c(_), Y = 3 ).

% Checked once, though switch detection puts it in a case.
% ERR-DAG: requires.m:[[@LINE+3]]:3: error: the goal in this scope is required to be det, but its determinism is semidet
% ERR-DAG: requires.m:[[@LINE+2]]:17: because: the call to >(in, in) can fail
in_cases(X, Y) :- ( X = a, Y = 1 ; X = b, Y = 2 ; X = c(Z),
  require_det ( Z > 0, Y = Z ) ).

% ERR-DAG: requires.m:[[@LINE+3]]:25: error: the lambda expression is declared det, but its determinism is inferred to be semidet
% ERR-DAG: requires.m:[[@LINE+3]]:5: because: the switch on X does not cover c/1
% ERR-DAG: requires.m:[[@LINE+1]]:48: error: the switch on X is required to be complete, but it does not cover c/1
in_lambda(X, Y) :- P = (pred(Z::out) is det :- require_complete_switch [X]
  ( X = a, Z = 1 ; X = b, Z = 2 )), P(Y).

% ERR-DAG: requires.m:[[@LINE+1]]:23: error: the switch on Z is required to be complete, but it does not cover c/1
in_clause(a, Z, Y) :- require_complete_switch [Z] ( Z = a, Y = 1 ;
  Z = b, Y = 2 ).
in_clause(b, _, 3).
