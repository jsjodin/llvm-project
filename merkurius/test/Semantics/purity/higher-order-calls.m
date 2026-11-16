% Higher-order calls (reference manual, "Purity annotations on
% higher-order calls"): a call's annotation is the purity of the closure
% it needs, so a missing, unnecessary or too weak one is a type error, as
% in mmc; apply and F(X) apply pure functions, and impure_apply and
% semipure_apply impure and semipure ones. A closure of an impure
% predicate is made purely, and has an impure type.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module higher_order_calls.
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

% Well-typed.
:- impure pred calls((impure pred(int))::in(pred(in) is det)) is det.
calls(P) :- impure P(1), impure call(P, 2).
:- semipure pred semipure_call((semipure pred(int))::in(pred(out) is det)) is det.
semipure_call(P) :- semipure P(_).
:- impure func applied(impure func(int) = int) = int.
applied(F) = Y :- impure Y = impure_apply(F, 1).
:- semipure func semipure_applied(semipure func(int) = int) = int.
semipure_applied(F) = Y :- semipure Y = semipure_apply(F, 1).
:- impure pred closure is det.
closure :- P = ip, impure P(1).
:- impure pred function_closure(int::out) is det.
function_closure(Y) :- F = ifn, impure Y = impure_apply(F, 1).

% Type errors.
:- impure pred missing((impure pred(int))::in(pred(in) is det)) is det.
missing(P) :- P(1).
% CHECK: [[@LINE-1]]:15: error: type error in higher-order call: variable P has type impure pred(int), but the call needs pred(int)
:- pred unnecessary((pred(int))::in(pred(in) is det)) is det.
unnecessary(P) :- impure P(1).
% CHECK: [[@LINE-1]]:26: error: type error in higher-order call: variable P has type pred(int), but the call needs impure pred(int)
:- impure pred too_weak((impure pred(int))::in(pred(in) is det)) is det.
too_weak(P) :- semipure P(1).
% CHECK: [[@LINE-1]]:25: error: type error in higher-order call: variable P has type impure pred(int), but the call needs semipure pred(int)
:- impure func plain_apply(impure func(int) = int) = int.
plain_apply(F) = Y :- impure Y = apply(F, 1).
% CHECK: [[@LINE-1]]:34: error: type error in higher-order call: variable F has type (impure func(int) = int), but the call needs (func(int) = int)
:- impure func plain_syntax(impure func(int) = int) = int.
plain_syntax(F) = Y :- impure Y = F(1).
% CHECK: [[@LINE-1]]:35: error: type error in higher-order call: variable F has type (impure func(int) = int), but the call needs (func(int) = int)
:- func impure_apply_of_pure(func(int) = int) = int.
impure_apply_of_pure(F) = Y :- impure Y = impure_apply(F, 1).
% CHECK: [[@LINE-1]]:43: error: type error in higher-order call: variable F has type (func(int) = int), but the call needs (impure func(int) = int)
