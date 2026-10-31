% Type errors with existentially typed constructors (reference manual,
% "Existentially typed data types"): a construction is written 'new f',
% only with an existentially typed constructor, and satisfies its
% constraints; a deconstruction's types cannot be bound, and two of them
% differ; and a field whose existential type variable another field
% shares cannot be updated.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module constructor_errors.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, string.

main(0).
:- typeclass showable(T) where [ func show(T) = string ].
:- instance showable(int) where [ show(X) = int_to_string(X) ].
:- type box ---> some [T] (box(T) => showable(T)).
:- type any ---> some [T] any(T).
:- type plain ---> plain(int).
:- type two ---> some [T] two(first :: T, second :: T).
:- type record ---> some [T] (record(item :: T, count :: int) => showable(T)).

:- func without_new = box.
without_new = box(1).
% CHECK: [[@LINE-1]]:19: error: type error: `1' has type int, but must have type (some [T] T) here
:- func new_plain = plain.
new_plain = 'new plain'(1).
% CHECK: [[@LINE-1]]:13: error: undefined symbol new plain/1
% CHECK: [[@LINE-2]]:13: because: `new' is only for existentially typed constructors, and plain/1 of type constructor_errors.plain/0 is not one
:- func unsatisfied = box.
unsatisfied = 'new box'("s").
% CHECK: [[@LINE-1]]:15: error: unsatisfiable type class constraint `constructor_errors.showable(string)'
:- func bound(box) = int.
bound(box(X)) = X.
% CHECK: [[@LINE-1]]:7: error: type error: `box(X)' must have type constructor_errors.box here, which no declaration of box/1 fits
% CHECK: [[@LINE-2]]:7: because: its arguments have types int
% CHECK: [[@LINE-3]]:7: because: it may be some [T] constructor constructor_errors.box(T) = constructor_errors.box => (constructor_errors.showable(T))
:- pred same(any::in, any::in) is semidet.
same(any(X), any(Y)) :- X = Y.
% CHECK: [[@LINE-1]]:25: error: type error in unification of variable X and variable Y: variable X has type (some [T] T), but variable Y has type (some [T] T)
:- func shared(two) = two.
shared(R) = R ^ first := 1.
% CHECK: [[@LINE-1]]:13: error: type error: `R ^ first := 1' must have type constructor_errors.two here, which no declaration of first :=/2 fits
% CHECK: [[@LINE-2]]:13: because: its arguments have types constructor_errors.two, T_2
% CHECK: [[@LINE-3]]:17: because: it may be field update constructor_errors.'first :='(constructor_errors.two, T) = constructor_errors.two
% CHECK: [[@LINE-4]]:17: because: field first cannot be updated, since its existentially quantified type variable T occurs in another field's type too
% CHECK: [[@LINE-27]]:31: because: the field is defined here
:- func replaced(record) = record.
replaced(R) = R ^ item := "s".
% CHECK: [[@LINE-1]]:19: error: unsatisfiable type class constraint `constructor_errors.showable(string)'
:- func selected(record) = int.
selected(R) = R ^ item.
% CHECK: [[@LINE-1]]:15: error: type error: `R ^ item' must have type int here, which no declaration of item/1 fits
% CHECK: [[@LINE-2]]:15: because: its arguments have types constructor_errors.record
% CHECK: [[@LINE-3]]:19: because: it may be some [T] field constructor_errors.item(constructor_errors.record) = T => (constructor_errors.showable(T))
