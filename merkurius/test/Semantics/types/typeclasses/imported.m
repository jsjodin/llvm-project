% Instances of imported type classes: their methods are visible only where
% the type class's definition is, and they must not overlap the imported
% instances.
%
% RUN: not merkc -fc1 -fdebug-dump-types -I %S/Inputs %s 2>&1 | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines --include %S/Inputs %s %}

:- module imported.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module classes, list.

:- type t ---> t.
main(0).
:- instance shown(t) where [ shown(_) = "t" ].
% mmc reports this only when linking (see the design document's register).
:- instance shown(int) where [ shown(_) = "int" ]. % mmc-differs
% CHECK: [[@LINE-1]]:13: error: instance classes.shown(int) overlaps an earlier instance of typeclass shown/1
:- instance shown(list(A)) where [ shown(_) = "list" ].
% CHECK: [[@LINE-1]]:13: error: instance classes.shown(list.list(A)) overlaps an earlier instance of typeclass shown/1
:- instance hidden(t) where [ hide(_) = 0 ].
% CHECK: [[@LINE-1]]:13: error: the methods of typeclass hidden/1 are not visible here, where its declaration is abstract
