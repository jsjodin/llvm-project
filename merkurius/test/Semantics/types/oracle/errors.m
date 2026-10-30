% A type error of each kind of goal, in declared and in undeclared
% predicates and functions. Each is reported once, at its goal, and the
% clause's other goals are still checked. mmc reports the errors of its
% first pass of inference with any, which these include.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --allow-errors %s %}

:- module errors.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, list, maybe, string.

:- type point ---> point(x :: int, y :: int).

main(0).
:- pred var_var(int::in, string::out) is det.
var_var(X, Y) :- Y = X.
% CHECK: errors.m:[[@LINE-1]]:18: error: type error in unification of variable Y and variable X: variable Y has type string, but variable X has type int
var_functor(X) :- X = 1, X = "a".
% CHECK: errors.m:[[@LINE-1]]:30: error: type error in unification of variable X and `"a"': variable X has type int, but `"a"' has type string
nested_functor(X) :- X = [1, "a"].
% CHECK: errors.m:[[@LINE-1]]:30: error: type error: `"a"' has type string, but must have type int here
call_argument(X) :- X = "a", list.length(X, _).
% CHECK: errors.m:[[@LINE-1]]:30: error: type error in call to length/2: its arguments have types string, T_1
% CHECK: errors.m:[[@LINE-2]]:30: because: it may be pred list.length(list.list(_T), int)
call_result(N) :- N = "a", N = list.length([1]).
% CHECK: errors.m:[[@LINE-1]]:32: error: type error in unification of variable N and `list.length([1])': variable N has type string, which no declaration of length/1 fits
% CHECK: errors.m:[[@LINE-2]]:32: because: its arguments have types T_1
% CHECK: errors.m:[[@LINE-3]]:32: because: it may be func list.length(list.list(T)) = int
% CHECK: errors.m:[[@LINE-4]]:32: because: it may be pred list.length(list.list(_T), int)
:- pred declared_call(string::in) is det.
declared_call(S) :- string.length(S, L), L = "x".
% CHECK: errors.m:[[@LINE-1]]:46: error: type error in unification of variable L and `"x"': variable L has type int, but `"x"' has type string
inferred_callee(X) :- X = 1, string_only(X).
% CHECK: errors.m:[[@LINE-1]]:30: error: type error in call to string_only/1: its arguments have types int
% CHECK: errors.m:[[@LINE-2]]:30: because: it may be pred errors.string_only(string)
string_only(S) :- S = "s".
ho_call(F, Y) :- F = (func(X) = X + 1), Y = F("a").
% CHECK: errors.m:[[@LINE-1]]:45: error: type error in higher-order call: variable F has type (func(int) = int), but the call needs (func(string) = T1)
ho_pred_call(P) :- P = (pred(X::in) is semidet :- X = 1), P("a").
% CHECK: errors.m:[[@LINE-1]]:59: error: type error in higher-order call: variable P has type pred(int), but the call needs pred(string)
wrong_arity_call(P) :- P = (pred(X::out) is det :- X = 1), call(P, _, _).
% CHECK: errors.m:[[@LINE-1]]:60: error: type error in higher-order call: variable P has type pred(int), but the call needs pred(T_2, T_3)
lambda_body(F) :- F = (func(X) = Y :- Y = X ++ "a", X = 1).
% CHECK: errors.m:[[@LINE-1]]:57: error: type error in unification of variable X and `1': variable X has type string, but `1' has type int
ite_branches(X, Y) :- ( if X = 1 then Y = 1 else Y = "a" ).
% CHECK: errors.m:[[@LINE-1]]:54: error: type error in unification of variable Y and `"a"': variable Y has type int, but `"a"' has type string
disj_branches(Y) :- ( Y = 1 ; Y = "a" ).
% CHECK: errors.m:[[@LINE-1]]:35: error: type error in unification of variable Y and `"a"': variable Y has type int, but `"a"' has type string
negation(X) :- X = 1, not X = "a".
% CHECK: errors.m:[[@LINE-1]]:31: error: type error in unification of variable X and `"a"': variable X has type int, but `"a"' has type string
qualification(X) :- X = 1, X = "a" : string.
% CHECK: errors.m:[[@LINE-1]]:25: error: type error in unification of variable X and `1': variable X has type string, but `1' has type int
field_access(P, X) :- P = point(1, 2), X = P ^ x, X = "a".
% CHECK: errors.m:[[@LINE-1]]:55: error: type error in unification of variable X and `"a"': variable X has type int, but `"a"' has type string
field_update(P, Q) :- P = point(1, 2), Q = P ^ y := "a".
% CHECK: errors.m:[[@LINE-1]]:53: error: type error: `"a"' has type string, but must have type int here
undefined_field(P, X) :- P = point(1, 2), X = P ^ z.
% CHECK: errors.m:[[@LINE-1]]:51: error: undefined symbol z/1
unknown_constructor(X) :- X = yes(1), X = no(2).
% CHECK: errors.m:[[@LINE-1]]:43: error: undefined symbol no/1
% CHECK: errors.m:[[@LINE-2]]:43: because: there is constructor maybe.no/0
constructor_closure(Xs) = map(yes, Xs).
% CHECK: errors.m:[[@LINE-1]]:31: error: undefined symbol yes/0
% CHECK: errors.m:[[@LINE-2]]:31: because: there is constructor maybe.yes/1
occurs(X) :- X = [X].
% CHECK: errors.m:[[@LINE-1]]:18: error: type error in unification of variable X and `[X]': variable X has type T1, which no declaration of [|]/2 fits
% CHECK: errors.m:[[@LINE-2]]:18: because: its arguments have types T1, T_2
% CHECK: errors.m:[[@LINE-3]]:18: because: it may be constructor list.'[|]'(T, list.list(T)) = list.list(T)
tuple_arity(T) :- T = {1, 2}, T = {1, 2, 3}.
% CHECK: errors.m:[[@LINE-1]]:35: error: type error in unification of variable T and `{1, 2, 3}': variable T has type {int, int}, but `{1, 2, 3}' has type {T_3, T_4, T_5}
head_mismatch([], 0).
head_mismatch("a", 1).
% CHECK: errors.m:[[@LINE-1]]:15: error: type error: `"a"' has type string, but must have type list.list(T1) here
:- func declared_result(int) = string.
declared_result(X) = X + 1.
% CHECK: errors.m:[[@LINE-1]]:22: error: type error: `X + 1' must have type string here, which no declaration of +/2 fits
% CHECK: errors.m:[[@LINE-2]]:22: because: its arguments have types int, T_2
% CHECK: errors.m:[[@LINE-3]]:24: because: it may be func int.'+'(int, int) = int
state_variables(!S) :- !:S = !.S + 1, !:S = !.S ++ "a".
% CHECK: errors.m:[[@LINE-1]]:45: error: type error in unification of state variable !S and `!.S ++ "a"': state variable !S has type T1, which no declaration of ++/2 fits
% CHECK: errors.m:[[@LINE-2]]:45: because: its arguments have types int, T_5
% CHECK: errors.m:[[@LINE-3]]:49: because: it may be func list.'++'(list.list(T), list.list(T)) = list.list(T)
% CHECK: errors.m:[[@LINE-4]]:49: because: it may be func string.'++'(string, string) = string
