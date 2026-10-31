% Purity annotations on calls and function applications (reference
% manual, "Marking a goal as impure"): a call of an impure or semipure
% predicate is annotated so, and an impure or semipure function is applied
% in a unification of its own, so annotated; a nested application of a
% pure function is fine. A missing annotation, or `semipure' where
% `impure' is needed, is an error; one more impure than the callee is a
% warning. An annotation is only for a call or a unification that applies
% a function.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module annotations.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int.

main(0).
:- impure pred ip(int::in) is det.
:- pragma foreign_proc("C", ip(X::in), [will_not_call_mercury], "(void) X;").
:- semipure pred sp(int::out) is det.
:- pragma foreign_proc("C", sp(X::out), [will_not_call_mercury, promise_semipure], "X = 0;").
:- impure func ifn(int) = int.
:- pragma foreign_proc("C", ifn(X::in) = (Y::out), [will_not_call_mercury], "Y = X;").
:- semipure func sfn(int) = int.
:- pragma foreign_proc("C", sfn(X::in) = (Y::out), [will_not_call_mercury, promise_semipure], "Y = X;").
:- pred pp(int::in) is det.
pp(_).
:- func pf(int) = int.
pf(X) = X.
:- type t ---> k(int).

% Annotated as needed.
:- impure pred annotated(int::out) is det.
annotated(Z) :-
    impure ip(1),
    semipure sp(X),
    impure ip(X),
    impure Y = ifn(pf(X)),
    semipure Z = sfn(Y).

% Missing, or not impure enough.
:- impure pred missing is det.
missing :- ip(1).
% CHECK: [[@LINE-1]]:12: error: the call of impure pred annotations.ip/1 must be annotated `impure'
:- semipure pred missing_semipure(int::out) is det.
missing_semipure(X) :- sp(X).
% CHECK: [[@LINE-1]]:24: error: the call of semipure pred annotations.sp/1 must be annotated `semipure'
:- impure pred semipure_for_impure is det.
semipure_for_impure :- semipure ip(1).
% CHECK: [[@LINE-1]]:24: error: the call of impure pred annotations.ip/1 must be annotated `impure'
:- impure pred function_unannotated(int::out) is det.
function_unannotated(Y) :- Y = ifn(1).
% CHECK: [[@LINE-1]]:32: error: impure func annotations.ifn/1 must be applied in a unification of its own, annotated `impure'
:- impure pred function_nested(int::out) is det.
function_nested(Y) :- impure Y = ifn(ifn(1)).
% CHECK: [[@LINE-1]]:38: error: impure func annotations.ifn/1 must be applied in a unification of its own, annotated `impure'
:- impure pred function_argument is det.
function_argument :- impure ip(ifn(1)).
% CHECK: [[@LINE-1]]:32: error: impure func annotations.ifn/1 must be applied in a unification of its own, annotated `impure'
:- impure pred semipure_nested(int::out) is det.
semipure_nested(Y) :- impure Y = ifn(sfn(1)).
% CHECK: [[@LINE-1]]:38: error: semipure func annotations.sfn/1 must be applied in a unification of its own, annotated `semipure'

% More impure than needed.
:- semipure pred impure_for_semipure(int::out) is det.
impure_for_semipure(X) :- impure sp(X).
% CHECK: [[@LINE-1]]:27: warning: the `impure' annotation is unnecessary, since pred annotations.sp/1 is semipure
:- pred impure_for_pure is det.
impure_for_pure :- impure pp(1).
% CHECK: [[@LINE-1]]:20: warning: the `impure' annotation is unnecessary, since pred annotations.pp/1 is pure
:- pred semipure_for_pure is det.
semipure_for_pure :- semipure pp(1).
% CHECK: [[@LINE-1]]:22: warning: the `semipure' annotation is unnecessary, since pred annotations.pp/1 is pure
:- pred impure_pure_function(int::out) is det.
impure_pure_function(X) :- impure X = pf(1).
% CHECK: [[@LINE-1]]:28: warning: the `impure' annotation is unnecessary, since func annotations.pf/1 is pure

% Not a unification that applies a function (for compound goals, see
% compound.m).
:- pred constructor(t::out) is det.
constructor(X) :- impure X = k(1).
% CHECK: [[@LINE-1]]:19: error: `impure' annotates a unification that applies no function
:- pred literal(int::out) is det.
literal(X) :- semipure X = 1.
% CHECK: [[@LINE-1]]:15: error: `semipure' annotates a unification that applies no function
% Between variables, mmc says nothing.
:- pred variables(int::in, int::out) is det.
variables(Y, X) :- impure X = Y.
