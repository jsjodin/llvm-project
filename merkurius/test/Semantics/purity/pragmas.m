% A purity pragma promises the declared purity, of a predicate or function
% the module declares, a function by its arity without its result
% (reference manual, "Promising that a predicate is pure"). For a pure
% declaration promised semipure, mmc says "declared pure but promised
% pure" (see the register).
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module pragmas.
:- interface.
:- pred main(int::out) is det.
:- implementation.

main(0).
:- impure pred ip(int::in) is det.
:- pragma foreign_proc("C", ip(X::in), [will_not_call_mercury], "(void) X;").
:- pred pure_promised_semipure is det.
% CHECK: [[@LINE-1]]:9: error: pred pragmas.pure_promised_semipure/0 is declared pure, but promised semipure
:- pragma promise_semipure(pure_promised_semipure/0).
pure_promised_semipure :- impure ip(1).
:- func by_result_arity(int) = int.
:- pragma promise_pure(by_result_arity/2).
% CHECK: [[@LINE-1]]:4: error: `:- pragma promise_pure' names by_result_arity/2, which this module does not declare; its arity is 1
by_result_arity(X) = X.
:- pragma promise_pure(nowhere/1).
% CHECK: [[@LINE-1]]:4: error: `:- pragma promise_pure' names nowhere/1, which this module does not declare
