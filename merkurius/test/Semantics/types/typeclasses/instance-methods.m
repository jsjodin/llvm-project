% The methods of an instance (reference manual, "Instance declarations"):
% each defined once, by clauses or by naming a predicate or function, and
% each a method of the type class.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module instance_methods.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module list.

:- typeclass c(T) where [ func f(T) = int, pred g(T::in) is semidet ].
:- type p(A, B) ---> p(A, B).
:- type q(A) ---> q(A).
:- type r(A) ---> r(A).

main(0).
:- instance c(int) where [ f(X) = X ].
% CHECK: [[@LINE-1]]:13: error: instance instance_methods.c(int) has no implementation of the predicate method g/1
:- instance c(string) where [ f(_) = 0, (g(_) :- true), h(_) = 1 ].
% CHECK: [[@LINE-1]]:57: error: the typeclass c/1 has no function method h/1
:- instance c(list(T)) where [ f(_) = 0, func(f/1) is ff, (g(_) :- true) ].
% CHECK: [[@LINE-1]]:42: error: instance instance_methods.c(list.list(T)) has more than one implementation of the function method f/1
% CHECK: [[@LINE-2]]:32: because: the first is here
:- instance c({}) where [ func(f/1) is ff, func(f/1) is ff, pred(g/1) is gg ].
% CHECK: [[@LINE-1]]:44: error: instance instance_methods.c({}) has more than one implementation of the function method f/1
% CHECK: [[@LINE-2]]:27: because: the first is here
:- instance c(q(T)) where [ pred(f/1) is gg, pred(g/1) is gg ].
% CHECK: [[@LINE-1]]:13: error: instance instance_methods.c(instance_methods.q(T)) has no implementation of the function method f/1
% CHECK: [[@LINE-2]]:29: error: the typeclass c/1 has no predicate method f/1
:- instance c(r(T)) where [ func(f/2) is ff, (g(_) :- true) ].
% CHECK: [[@LINE-1]]:13: error: instance instance_methods.c(instance_methods.r(T)) has no implementation of the function method f/1
% CHECK: [[@LINE-2]]:29: error: the typeclass c/1 has no function method f/2
% Allowed: several clauses for a method.
:- instance c(p(A, B)) where [ f(_) = 0, f(_) = 1, (g(_) :- true) ].
:- func ff(T) = int.
ff(_) = 0.
:- pred gg(T::in) is semidet.
gg(_).
