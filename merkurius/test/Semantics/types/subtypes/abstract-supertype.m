% An abstract type's definition is not in scope, so it cannot be a
% supertype (reference manual, "Subtypes": the supertype's definition
% must be "in scope where the subtype definition occurs, by normal module
% visibility rules"). mmc accepts it for an abstract type that is not a
% subtype, checking the constructors against the definition the module
% cannot see, so it is not compared (see the register).
%
% RUN: not merkc -fc1 -fdebug-dump-types -I %S/Inputs %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s

:- module abstract_supertype.
:- interface.
:- import_module fruits.
:- type public =< secret ---> secret.
% CHECK: [[@LINE-1]]:4: error: the supertype fruits.secret of public/0 is not a discriminated union type whose definition is visible
:- implementation.
