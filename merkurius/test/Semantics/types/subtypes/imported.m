% Subtypes of imported types: an imported subtype is a supertype like any
% other, and an abstractly exported subtype has no definition in scope, so
% it cannot be a supertype (reference manual, "Subtypes"). For an abstract
% type that is not a subtype, see abstract-supertype.m.
%
% RUN: not merkc -fc1 -fdebug-dump-types -I %S/Inputs %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines --include %S/Inputs %s %}

:- module imported.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module fruits.

main(0).
:- type lemon_only =< citrus ---> lemon.
:- type not_citrus =< citrus ---> lemon ; pear.
% CHECK: [[@LINE-1]]:43: error: pear/0 is not a constructor of the supertype fruits.citrus
:- type apples =< pome ---> apple.
% CHECK: [[@LINE-1]]:4: error: the supertype fruits.pome of apples/0 is not a discriminated union type whose definition is visible
