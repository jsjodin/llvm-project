% Type errors in instances' methods: a clause that does not fit the
% method's type on the instance's types, a definition by name whose
% predicate or function is undefined, of the other kind, or not general
% enough, and constraints that neither the instance nor the method gives,
% and a method's own type variable, which is rigid.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module method_errors.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module float, int, list, string.

:- typeclass show(T) where [
    func show(T) = string,
    pred show_to(T::in, string::out) is det
].
:- typeclass conv(A, B) where [
    func conv(A) = B,
    func with_shown(A, B, C) = string <= show(C)
].

main(0).

:- instance show(int) where [
    show(X) = X,
% CHECK: [[@LINE-1]]:15: error: type error in unification of `X' and variable X: `X' has type string, but variable X has type int
    pred(show_to/2) is no_such_pred
% CHECK: [[@LINE-1]]:24: error: undefined predicate no_such_pred/2
].

:- instance show(float) where [
    func(show/1) is no_such_func,
% CHECK: [[@LINE-1]]:21: error: undefined symbol no_such_func/1
    (show_to(X, S) :- S = X)
% CHECK: [[@LINE-1]]:23: error: type error in unification of variable S and variable X: variable S has type string, but variable X has type float
].

% A predicate where a function is needed is a closure, and a function
% where a predicate is needed is not a predicate.
:- instance show(string) where [
    func(show/1) is to_empty,
% CHECK: [[@LINE-1]]:21: error: type error: `to_empty' must have type string here, which no declaration of to_empty/1 fits
% CHECK: [[@LINE-2]]:21: because: its arguments have types string
% CHECK: [[@LINE-3]]:21: because: it may be pred method_errors.to_empty(string, string)
    pred(show_to/2) is from_int
% CHECK: [[@LINE-1]]:24: error: undefined predicate from_int/2
].
:- pred to_empty(string::in, string::out) is det.
to_empty(_, "").
:- func from_int(string) = string.
from_int(S) = S.

% The definition's type must be at least as general as the method's.
:- instance show(list(T)) where [
    func(show/1) is ints_shown,
% CHECK: [[@LINE-1]]:21: error: type error: `ints_shown' must have type string here, which no declaration of ints_shown/1 fits
% CHECK: [[@LINE-2]]:21: because: its arguments have types list.list(T)
% CHECK: [[@LINE-3]]:21: because: it may be func method_errors.ints_shown(list.list(int)) = string
    pred(show_to/2) is shown_to
% CHECK: [[@LINE-1]]:24: error: unsatisfiable type class constraint `method_errors.show(T)'
].
:- func ints_shown(list(int)) = string.
ints_shown(_) = "".
:- pred shown_to(list(T)::in, string::out) is det <= show(T).
shown_to(Xs, show(Xs)).

% No error: show(list(T)) has an instance without constraints.
:- instance conv(int, string) where [
    conv(X) = show([X]),
    with_shown(_, _, Y) = show([Y])
].

:- instance conv(string, int) where [
    conv(S) = string.length(S),
    with_shown(_, _, Y) = show(Y ++ "")
% CHECK: [[@LINE-1]]:32: error: type error: `Y ++ ""' must have type T_7 here, which no declaration of ++/2 fits
% CHECK: [[@LINE-2]]:32: because: its arguments have types C, T_6
% CHECK: [[@LINE-3]]:34: because: it may be func list.'++'(list.list(T), list.list(T)) = list.list(T)
% CHECK: [[@LINE-4]]:34: because: it may be func string.'++'(string, string) = string
].
