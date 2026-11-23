% Type conversions to a supertype of another module (reference manual,
% "Subtypes", "Type conversions"). A subtype's constructors are of its own
% module, its supertype's of theirs; they correspond by name and arity, and
% the result's functors are its type's. A bound inst and a ground one both
% convert.
%
% RUN: merkc -fc1 -fdebug-dump-modes %s | FileCheck %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module coerce_imported.
:- interface.
:- import_module list, pair.

:- type nel(T) =< list(T) ---> [T | list(T)].
:- type same =< pair(int, int) ---> int - int.

:- func bound_up = list(int).
:- func ground_up(nel(int)) = list(int).
:- func pair_up = pair(int, int).

:- implementation.

bound_up = coerce(X) :-
    X = [1] : nel(int).
% CHECK-LABEL: func coerce_imported.bound_up/0 mode 0: bound_up = out is det
% CHECK: final: {{.*}}HeadVar__1 unique('[|]'(unique(1), unique([])))

ground_up(X) = coerce(X).
% CHECK-LABEL: func coerce_imported.ground_up/1 mode 0: ground_up(in) = out is det
% CHECK: final: X ground, HeadVar__2 bound('[|]'(ground, ground))

pair_up = coerce(X) :-
    X = (3 - 4) : same.
% CHECK-LABEL: func coerce_imported.pair_up/0 mode 0: pair_up = out is det
% CHECK: final: {{.*}}HeadVar__1 unique('-'(unique(3), unique(4)))
