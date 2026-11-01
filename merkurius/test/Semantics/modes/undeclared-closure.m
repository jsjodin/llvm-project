% A higher-order term of a predicate or function with no declared
% determinism is an error, as in mmc, which makes such a term a lambda
% expression with its callee's declared determinism before determinism
% inference.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module undeclared_closure.
:- interface.

:- pred closure(int::out) is det.
:- pred function_closure(int::out) is semidet.

:- implementation.
:- import_module int.

:- pred undeclared(int::in, int::out).
undeclared(X, X).
:- func undeclared_f(int, int) = int.
:- mode undeclared_f(in, in) = out.
undeclared_f(X, Y) = X + Y.

closure(Y) :-
    P = undeclared(1),
% CHECK: [[@LINE-1]]:9: error: in the clause for closure(out): the pred undeclared_closure.undeclared/2 has no declared determinism, so a curried call to it may not be used as a lambda expression
    P(Y).
function_closure(Y) :-
    F = undeclared_f(1),
% CHECK: [[@LINE-1]]:9: error: in the clause for function_closure(out): the func undeclared_closure.undeclared_f/2 has no declared determinism, so a curried call to it may not be used as a lambda expression
    Y = F(2).
