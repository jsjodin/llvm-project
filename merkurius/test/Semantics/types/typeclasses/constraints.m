% Constraints of calls satisfied (thesis, section 3.1.4): by the
% declaration's constraints, directly or through a superclass, or by an
% instance whose own constraints are satisfied in turn. The dump shows how
% each is.
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types %s %}

:- module constraints.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module float, int, list, string.

:- typeclass show(T) where [ func show(T) = string ].
:- typeclass loud(T) <= show(T) where [ func loud(T) = string ].
:- typeclass conv(A, B) where [ func conv(A) = B ].
:- type w(T) ---> w(T).
:- instance show(int) where [ show(X) = string.int_to_string(X) ].
:- instance show(float) where [ show(X) = string.float_to_string(X) ].
:- instance show(list(T)) <= show(T) where [ show(_) = "[...]" ].
:- instance show({A, B}) <= (show(A), show(B)) where [
    show({X, Y}) = show(X) ++ show(Y)
].
:- instance show(w(T)) <= show(T) where [ show(w(X)) = show(X) ].
:- instance loud(int) where [ loud(X) = show(X) ++ "!" ].
:- instance loud(w(T)) <= loud(T) where [ loud(w(X)) = loud(X) ].
:- instance conv(int, string) where [ conv(X) = show(X) ].

main(0).
:- func by_instance = string.
by_instance = show(1).
% CHECK: clause at [[@LINE-1]]:1 of func constraints.by_instance/0
% CHECK:   constraint of show at [[@LINE-2]]:15: constraints.show(int) by the instance at 19:13
:- func by_premises = string.
by_premises = show([{1, [2.5]}]).
% CHECK: clause at [[@LINE-1]]:1 of func constraints.by_premises/0
% CHECK:   constraint of show at [[@LINE-2]]:15: constraints.show(list.list({int, list.list(float)})) by the instance at 21:13
% CHECK-NEXT:   constraints.show({int, list.list(float)}) by the instance at 22:13
% CHECK-NEXT:     constraints.show(int) by the instance at 19:13
% CHECK-NEXT:     constraints.show(list.list(float)) by the instance at 21:13
% CHECK-NEXT:       constraints.show(float) by the instance at 20:13
:- func assumed(T) = string <= show(T).
assumed(X) = show(X).
% CHECK: clause at [[@LINE-1]]:1 of func constraints.assumed/1
% CHECK:   constraint of show at [[@LINE-2]]:14: constraints.show(T) by the declaration's constraints.show(T)
:- func superclass(T) = string <= loud(T).
superclass(X) = show(X) ++ loud(X).
% CHECK: clause at [[@LINE-1]]:1 of func constraints.superclass/1
% CHECK:   constraint of show at [[@LINE-2]]:17: constraints.show(T) by the declaration's constraints.loud(T)
% CHECK:   constraint of loud at [[@LINE-3]]:28: constraints.loud(T) by the declaration's constraints.loud(T)
:- func mixed(T) = string <= loud(T).
mixed(X) = show(w([X])).
% CHECK: clause at [[@LINE-1]]:1 of func constraints.mixed/1
% CHECK:   constraint of show at [[@LINE-2]]:12: constraints.show(constraints.w(list.list(T))) by the instance at 25:13
% CHECK-NEXT:   constraints.show(list.list(T)) by the instance at 21:13
% CHECK-NEXT:     constraints.show(T) by the declaration's constraints.loud(T)
:- func two(A, B) = string <= (show(A), show(B)).
two(X, Y) = show({Y, X}).
% CHECK: clause at [[@LINE-1]]:1 of func constraints.two/2
% CHECK:   constraint of show at [[@LINE-2]]:13: constraints.show({B, A}) by the instance at 22:13
% CHECK-NEXT:   constraints.show(B) by the declaration's constraints.show(B)
% CHECK-NEXT:   constraints.show(A) by the declaration's constraints.show(A)
:- func multi = string.
multi = conv(1).
% CHECK: clause at [[@LINE-1]]:1 of func constraints.multi/0
% CHECK:   constraint of conv at [[@LINE-2]]:9: constraints.conv(int, string) by the instance at 28:13
:- func multi_assumed(A) = B <= conv(A, B).
multi_assumed(X) = conv(X).
% CHECK: clause at [[@LINE-1]]:1 of func constraints.multi_assumed/1
% CHECK:   constraint of conv at [[@LINE-2]]:20: constraints.conv(A, B) by the declaration's constraints.conv(A, B)
% A closure of a method, and a call in a lambda.
:- func closures = list(string).
closures = list.map(show, [1, 2]) ++ list.map(func(X) = show(X), [1.5]).
% CHECK: clause at [[@LINE-1]]:1 of func constraints.closures/0
% CHECK:   constraint of show at [[@LINE-2]]:21: constraints.show(int) by the instance at 19:13
% CHECK:   constraint of show at [[@LINE-3]]:57: constraints.show(float) by the instance at 20:13
% A ground constraint is assumed, and satisfied where it is called.
:- func ground(int) = string <= show(int).
ground(X) = show(X).
% CHECK: clause at [[@LINE-1]]:1 of func constraints.ground/1
% CHECK:   constraint of show at [[@LINE-2]]:13: constraints.show(int) by the declaration's constraints.show(int)
:- func calls_ground = string.
calls_ground = ground(1) ++ assumed(2) ++ superclass(3).
% CHECK: clause at [[@LINE-1]]:1 of func constraints.calls_ground/0
% CHECK:   constraint of ground at [[@LINE-2]]:16: constraints.show(int) by the instance at 19:13
% CHECK:   constraint of assumed at [[@LINE-3]]:29: constraints.show(int) by the instance at 19:13
% CHECK:   constraint of superclass at [[@LINE-4]]:43: constraints.loud(int) by the instance at 26:13
% An undeclared function's ground constraints.
inferred = show(w(1)).
% CHECK: clause at [[@LINE-1]]:1 of func constraints.inferred/0: inferred func constraints.inferred = string
% CHECK:   constraint of show at [[@LINE-2]]:12: constraints.show(constraints.w(int)) by the instance at 25:13
% CHECK-NEXT:   constraints.show(int) by the instance at 19:13
