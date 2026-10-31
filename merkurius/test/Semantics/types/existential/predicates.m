% Existentially typed predicates and functions (reference manual,
% "Existential types"): a callee determines its existentially quantified
% type variables, the same in each of its clauses, and each of its calls
% gives them types of its own, which the caller cannot bind and which
% satisfy the callee's existential constraints. The first examples are
% the manual's.
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types %s %}

:- module predicates.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, list, string, univ.

main(0).
:- typeclass showable(T) where [ func show(T) = string ].
:- instance showable(int) where [ show(X) = int_to_string(X) ].
:- instance showable(string) where [ show(S) = S ].

:- some [T] pred e_foo(T::out) is det.
e_foo(X) :- X = 42.
% CHECK: clause at [[@LINE-1]]:1 of pred predicates.e_foo/1: some [T] pred predicates.e_foo(T)
% CHECK-NEXT: X: int
:- pred call_e_foo is det.
call_e_foo :- e_foo(_).
% CHECK: clause at [[@LINE-1]]:1 of pred predicates.call_e_foo/0: pred predicates.call_e_foo

:- some [T] pred e_bar(T::out, T::out, (func(T) = int)::out(func(in) = out is det)) is det.
e_bar(2, 3, (func(X) = X * X)).
% CHECK: clause at [[@LINE-1]]:1 of pred predicates.e_bar/3: some [T] pred predicates.e_bar(T, T, (func(T) = int))
:- func call_e_bar = int.
call_e_bar = F(X) + F(Y) :- e_bar(X, Y, F).
% CHECK: clause at [[@LINE-1]]:1 of func predicates.call_e_bar/0: func predicates.call_e_bar = int
% CHECK-NEXT: F: (func((some [T] T)) = int)
% CHECK-NEXT: X: (some [T] T)
% CHECK-NEXT: Y: (some [T] T)

:- some [T] pred e_bar1(T::out) is multi.
e_bar1(42).
e_bar1(42).
e_bar1(43).
% CHECK: clause at [[@LINE-3]]:1 of pred predicates.e_bar1/1: some [T] pred predicates.e_bar1(T)
% CHECK: clause at [[@LINE-3]]:1 of pred predicates.e_bar1/1: some [T] pred predicates.e_bar1(T)
% CHECK: clause at [[@LINE-3]]:1 of pred predicates.e_bar1/1: some [T] pred predicates.e_bar1(T)

% An existential variable may be bound to a universal one, and a call's
% type may go into a polymorphic structure or to a polymorphic predicate.
:- some [U] pred hide(T::in, U::out) is det.
hide(X, X).
% CHECK: clause at [[@LINE-1]]:1 of pred predicates.hide/2: some [U] pred predicates.hide(T, U)
:- func length_of = int.
length_of = list.length([X, X]) :- e_foo(X).
% CHECK: clause at [[@LINE-1]]:1 of func predicates.length_of/0: func predicates.length_of = int
% CHECK-NEXT: X: (some [T] T)

% The callee satisfies its existential constraints, by instances or by
% its callees' existential constraints, and its callers may assume them.
:- some [T] func shown_value = T => showable(T).
shown_value = "s".
% CHECK: clause at [[@LINE-1]]:1 of func predicates.shown_value/0: some [T] func predicates.shown_value = T => (predicates.showable(T))
% CHECK-NEXT: existential constraint of the declaration: predicates.showable(string) by the instance at 20:13
:- func shown = string.
shown = show(shown_value).
% CHECK: clause at [[@LINE-1]]:1 of func predicates.shown/0: func predicates.shown = string
% CHECK: constraint of show at [[@LINE-2]]:9: predicates.showable((some [T] T)) by the existential constraint of shown_value at [[@LINE-2]]:14
:- some [T] pred passed_on(T::out) is det => showable(T).
passed_on(X) :- X = shown_value.
% CHECK: clause at [[@LINE-1]]:1 of pred predicates.passed_on/1: some [T] pred predicates.passed_on(T) => (predicates.showable(T))
% CHECK: existential constraint of the declaration: predicates.showable((some [T] T)) by the existential constraint of shown_value at [[@LINE-2]]:21

% univ's existential function gives a predicate values of different types
% (reference manual, "Some idioms using existentially quantified types").
:- some [T] pred good_example(string::in, T::out) is nondet.
good_example(Name, univ_value(Univ)) :-
    ( Name = "foo", Univ = univ(42)
    ; Name = "bar", Univ = univ("blah")
    ).
% CHECK: clause at [[@LINE-4]]:1 of pred predicates.good_example/2: some [T] pred predicates.good_example(string, T)
% CHECK: [[@LINE-5]]:20 univ_value -> func univ.univ_value/1

% A method may be existentially typed; each instance's definition
% determines its types.
:- typeclass box(B) where [ some [T] func unbox(B) = T => showable(T) ].
:- instance box(int) where [ unbox(X) = X ].
:- instance box(list(E)) where [ (unbox([]) = 1), (unbox([_ | _]) = 2) ].
:- func show_box(B) = string <= box(B).
show_box(B) = show(unbox(B)).
% CHECK: clause at [[@LINE-1]]:1 of func predicates.show_box/1: func predicates.show_box(B) = string <= (predicates.box(B))
% CHECK: constraint of show at [[@LINE-2]]:15: predicates.showable((some [T] T)) by the existential constraint of unbox at [[@LINE-2]]:20

% Inference gives an existential type only from a call's.
inferred(X) :- e_foo(X).
% CHECK: clause at [[@LINE-1]]:1 of pred predicates.inferred/1: inferred some [T] pred predicates.inferred(T)
in_list([X]) :- e_foo(X).
% CHECK: clause at [[@LINE-1]]:1 of pred predicates.in_list/1: inferred some [T] pred predicates.in_list(list.list(T))
constrained(X) :- passed_on(X).
% CHECK: clause at [[@LINE-1]]:1 of pred predicates.constrained/1: inferred some [T] pred predicates.constrained(T) => (predicates.showable(T))
chained(X) :- constrained(X).
% CHECK: clause at [[@LINE-1]]:1 of pred predicates.chained/1: inferred some [T] pred predicates.chained(T) => (predicates.showable(T))
in_closure(P) :- P = (pred(X::out) is det :- e_foo(X)).
% CHECK: clause at [[@LINE-1]]:1 of pred predicates.in_closure/1: inferred some [T] pred predicates.in_closure(pred(T))
uses_inferred = show(X) :- chained(X).
% CHECK: clause at [[@LINE-1]]:1 of func predicates.uses_inferred/0: inferred func predicates.uses_inferred = string
% CHECK: clause at [[@LINE-20]]:30 of func predicates.unbox/1 in instance predicates.box(int): some [T] func predicates.unbox(int) = T => (predicates.showable(T))
% CHECK: existential constraint of the declaration: predicates.showable(int) by the instance at 19:13
% CHECK: clause at [[@LINE-21]]:35 of func predicates.unbox/1 in instance predicates.box(list.list(E))
% CHECK: clause at [[@LINE-22]]:52 of func predicates.unbox/1 in instance predicates.box(list.list(E))
