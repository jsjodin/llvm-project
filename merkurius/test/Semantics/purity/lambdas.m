% A lambda expression's body is at most as impure as its annotation
% (reference manual, "Purity annotations on lambda expressions"); one
% annotated less pure than its body is fine, as in mmc. A lambda
% expression's body does not count towards its clause's purity, and its
% goals are not under the annotation of the goal it is in. impure_apply
% needs no annotation, as in mmc (see the register).
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module lambdas.
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
:- pred pp(int::in) is det.
pp(_).

:- impure pred annotated is det.
annotated :- P = (impure pred(X::in) is det :- impure ip(X)), impure P(1).
:- impure pred less_pure is det.
less_pure :- P = (impure pred(X::in) is det :- pp(X)), impure P(1).
:- pred unannotated(int::in) is det.
unannotated(Y) :- P = (pred(X::in) is det :- impure ip(X)), P(Y).
% CHECK: [[@LINE-1]]:24: error: the body of this lambda expression is impure, so it must be annotated `impure'
:- pred semipure_body(int::in) is det.
% CHECK: [[@LINE-1]]:9: error: pred lambdas.semipure_body/1 is semipure, so it must be declared `semipure' or promised pure
semipure_body(Y) :- P = (semipure pred(X::in) is det :- impure ip(X)), semipure P(Y).
% CHECK: [[@LINE-1]]:26: error: the body of this lambda expression is impure, so it must be annotated `impure'
:- pred semipure_for_pure(int::out) is det.
semipure_for_pure(Y) :- P = (pred(X::out) is det :- semipure sp(X)), P(Y).
% CHECK: [[@LINE-1]]:30: error: the body of this lambda expression is semipure, so it must be annotated `semipure'
:- impure func unannotated_apply(impure func(int) = int) = int.
unannotated_apply(F) = Y :- X = 1, Y = impure_apply(F, X).
