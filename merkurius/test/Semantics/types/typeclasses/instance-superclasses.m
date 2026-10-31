% An instance's type class's superclass constraints, on the instance's
% types, must follow from the instance's constraints and the other
% instances (thesis, section 3.1.4).
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module instance_superclasses.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module list, string.

:- typeclass show(T) where [ func show(T) = string ].
:- typeclass loud(T) <= show(T) where [ func loud(T) = string ].
:- typeclass pair(A, B) <= (show(A), show(B)) where [].
:- type w(T) ---> w(T).
:- type v(T) ---> v(T).
:- type u(T) ---> u(T).
:- instance show(int) where [ show(_) = "int" ].
:- instance show(w(T)) <= show(T) where [ show(_) = "w" ].
:- instance show(v(T)) <= show(T) where [ show(_) = "v" ].

main(0).
:- instance loud(int) where [ loud(_) = "INT" ].
:- instance loud(string) where [ loud(_) = "STRING" ].
% CHECK: [[@LINE-1]]:13: error: the superclass constraint `instance_superclasses.show(string)' of instance instance_superclasses.loud(string) is not satisfied
:- instance loud(list(T)) where [ loud(_) = "LIST" ].
% CHECK: [[@LINE-1]]:13: error: the superclass constraint `instance_superclasses.show(list.list(T))' of instance instance_superclasses.loud(list.list(T)) is not satisfied
% By the instance for w(T), whose show(T) follows from loud(T).
:- instance loud(w(T)) <= loud(T) where [ loud(_) = "W" ].
% The instance for v(T) needs show(T), which nothing gives.
:- instance loud(v(T)) where [ loud(_) = "V" ].
% CHECK: [[@LINE-1]]:13: error: the superclass constraint `instance_superclasses.show(instance_superclasses.v(T))' of instance instance_superclasses.loud(instance_superclasses.v(T)) is not satisfied
:- instance loud(u(T)) <= show(T) where [ loud(_) = "U" ].
% CHECK: [[@LINE-1]]:13: error: the superclass constraint `instance_superclasses.show(instance_superclasses.u(T))' of instance instance_superclasses.loud(instance_superclasses.u(T)) is not satisfied
:- instance pair(int, w(T)) <= show(T) where [].
:- instance pair(w(A), v(B)) <= show(A) where [].
% CHECK: [[@LINE-1]]:13: error: the superclass constraint `instance_superclasses.show(instance_superclasses.v(B))' of instance instance_superclasses.pair(instance_superclasses.w(A), instance_superclasses.v(B)) is not satisfied
