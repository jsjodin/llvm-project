% Type errors, where mmc reports them too. Each is reported at the term,
% and the goal is then skipped, so that a clause can show several.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck %s

:- module errors.
:- interface.
:- import_module list.
:- pred q(int::in, string::out) is det.
:- pred r(list(int)::out) is det.
:- pred s(int::out) is det.
:- pred t(int::in) is det.
:- pred u(pred(int)::in) is det.
:- implementation.
:- import_module int.

q(X, Y) :- Y = X.
% CHECK: errors.m:[[@LINE-1]]:12: error: type error in unification of variable Y and variable X: variable Y has type string, but variable X has type int
r(L) :- L = [1, "two"].
% CHECK: errors.m:[[@LINE-1]]:17: error: type error: `"two"' has type string, but must have type int here
s(X) :- X = 1.5 + 2.
% CHECK: errors.m:[[@LINE-1]]:13: error: type error: `1.5' has type float, but must have type int here
t(X) :- length(X, _), Y = X + "a", Y = Y.
% CHECK: errors.m:[[@LINE-1]]:9: error: type error in call to length/2: its arguments have types int, T_{{[0-9]+}}
% CHECK: because: it may be pred list.length(list.list(_T), int)
% CHECK: errors.m:[[@LINE-3]]:31: error: type error: `"a"' has type string, but must have type int here
u(P) :- P(1, 2).
% CHECK: errors.m:[[@LINE-1]]:9: error: type error in higher-order call: variable P has type pred(int), but the call needs pred(int, int)

% A goal with a type error has no declaration chosen.
% CHECK:      clause at 23:1 of pred errors.t/1: pred errors.t(int)
% CHECK-NEXT:   X: int
% CHECK-NEXT:   Y: int
% CHECK-NEXT:   23:29 + -> func int.'+'/2
