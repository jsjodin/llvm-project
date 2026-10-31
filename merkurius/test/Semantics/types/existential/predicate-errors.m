% Type errors with existentially typed predicates and functions (reference
% manual, "Examples of incorrect code using type quantifiers"): a caller
% cannot bind a call's existential types, nor unify those of two calls;
% a callee's clauses give its existential variables one type; and the
% callee satisfies its existential constraints, which its caller may
% assume, but no others. As in mmc, no closure of an existentially typed
% predicate can be made (see the register).
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module predicate_errors.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, list, string.

main(0).
:- typeclass showable(T) where [ func show(T) = string ].
:- instance showable(int) where [ show(X) = int_to_string(X) ].

:- pred bad_foo(T::in) is semidet.
bad_foo(42).
% CHECK: [[@LINE-1]]:9: error: type error: `42' has type int, but must have type T here

:- some [T] pred e_foo(T::out) is det.
e_foo(42).
:- pred bad_call_e_foo is semidet.
bad_call_e_foo :- e_foo(42).
% CHECK: [[@LINE-1]]:19: error: type error in call to e_foo/1: its arguments have types int
% CHECK: [[@LINE-2]]:19: because: it may be some [T] pred predicate_errors.e_foo(T)
:- pred bound_later is semidet.
bound_later :- e_foo(X), X = 1.
% CHECK: [[@LINE-1]]:30: error: type error in unification of variable X and `1': variable X has type (some [T] T), but `1' has type int
:- pred two_calls is semidet.
two_calls :- e_foo(X), e_foo(Y), X = Y.
% CHECK: [[@LINE-1]]:34: error: type error in unification of variable X and variable Y: variable X has type (some [T] T), but variable Y has type (some [T] T)

:- some [T] pred bad_e_bar2(T::out) is multi.
bad_e_bar2(42).
bad_e_bar2("blah").
% CHECK: [[@LINE-1]]:12: error: type error: `"blah"' has type string, but must have type int here
:- some [T] pred bad_e_bar3(T::out) is multi.
bad_e_bar3(X) :- e_foo(X).
bad_e_bar3(X) :- e_foo(X).
% CHECK: [[@LINE-1]]:18: error: type error in call to e_foo/1: its arguments have types (some [T] T)
% CHECK: [[@LINE-2]]:18: because: it may be some [T] pred predicate_errors.e_foo(T)
:- some [T] pred bad_example2(string::in, T::out) is semidet.
bad_example2(Name, Value) :-
    ( Name = "foo", Value = 42
    ; Name = "bar", Value = "blah"
% CHECK: [[@LINE-1]]:29: error: type error in unification of variable Value and `"blah"': variable Value has type int, but `"blah"' has type string
    ).

:- some [T] pred constrained(T::out) is det => showable(T).
constrained("x").
% CHECK: [[@LINE-1]]:1: error: unsatisfiable type class constraint `predicate_errors.showable(string)'
:- func not_given = string.
not_given = show(X) :- e_foo(X).
% CHECK: [[@LINE-1]]:13: error: unsatisfiable type class constraint `predicate_errors.showable((some [T] T))'

:- pred closure(pred(T)::out(pred(out) is det)) is det.
closure(P) :- P = e_foo.
% CHECK: [[@LINE-1]]:19: error: type error in unification of variable P and `e_foo': variable P has type pred(T), which no declaration of e_foo/0 fits
% CHECK: [[@LINE-2]]:19: because: it may be some [T] pred predicate_errors.e_foo(T)
% CHECK: [[@LINE-3]]:19: because: some [T] pred predicate_errors.e_foo(T) is existentially typed, and no closure of it can be made
