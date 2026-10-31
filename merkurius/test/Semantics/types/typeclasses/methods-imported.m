% The methods of an instance of an imported type class are checked by its
% methods' signatures, and may use the imported instances.
%
% RUN: merkc -fc1 -fdebug-dump-types -I %S/Inputs %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --include %S/Inputs %s %}

:- module methods_imported.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module classes, float, list, maybe, string.

main(0).
:- instance shown(float) where [ shown(F) = string.float_to_string(F) ].
% CHECK: clause at [[@LINE-1]]:34 of func classes.shown/1 in instance classes.shown(float): func classes.shown(float) = string
:- instance shown(maybe(T)) <= shown(T) where [
    shown(no) = "",
    shown(yes(X)) = shown([X])
% CHECK: clause at [[@LINE-1]]:5 of func classes.shown/1 in instance classes.shown(maybe.maybe(T)): func classes.shown(maybe.maybe(T)) = string <= (classes.shown(T))
% CHECK: constraint of shown at [[@LINE-2]]:21: classes.shown(list.list(T)) by the instance at classes.m:9:13
% CHECK-NEXT: classes.shown(T) by the declaration's classes.shown(T)
].
