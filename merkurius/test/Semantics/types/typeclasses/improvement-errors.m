% Where an instance's range does not fit a constraint's, the constraint is
% not improved, and since no instance gives it, it is unsatisfiable, as
% mmc reports it. Improvement does not choose among overloadings: of
% string.length and list.length, only one fits baz(int)'s range, but the
% call is ambiguous, as in mmc.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module improvement_errors.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, list, string.

main(0).
:- typeclass baz(A, B) <= (A -> B) where [ func baz(A) = B ].
:- instance baz(int, string) where [ baz(X) = string.int_to_string(X) ].
:- instance baz(list(S), list(T)) <= baz(S, T) where [ baz(_) = [] ].

:- func declared_int = int.
declared_int = baz(1).
% CHECK: [[@LINE-1]]:16: error: unsatisfiable type class constraint `improvement_errors.baz(int, int)'
:- func declared_list = list(int).
declared_list = baz([1]).
% CHECK: [[@LINE-1]]:17: error: unsatisfiable type class constraint `improvement_errors.baz(int, int)'
% CHECK: [[@LINE-2]]:17: because: an instance for `improvement_errors.baz(list.list(int), list.list(int))' needs it
lengths = length(baz(1)).
% CHECK: [[@LINE-1]]:1: error: ambiguous overloading causes type ambiguity
% CHECK: [[@LINE-2]]:11: because: length may be func list.length/1 or pred list.length/2 or func string.length/1 or pred string.length/2
