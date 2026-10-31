% An imported abstract type is not a discriminated union type whose
% definition is visible, so it cannot be coerced (reference manual, "Type
% conversions"), even to itself. mmc crashes on these ("Unexpected: not
% du type"), so they are not compared (see the register).
%
% RUN: not merkc -fc1 -fdebug-dump-types -I %S/Inputs %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s

:- module abstract.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module fruits.

main(0).
:- func to_fruit(secret(citrus)) = secret(fruit).
to_fruit(X) = coerce(X).
% CHECK: [[@LINE-1]]:15: error: cannot coerce from fruits.secret(fruits.citrus) to fruits.secret(fruits.fruit)
% CHECK: [[@LINE-2]]:15: because: fruits.secret(fruits.citrus) is not a discriminated union type whose definition is visible
:- func to_itself(secret(citrus)) = secret(citrus).
to_itself(X) = coerce(X).
% CHECK: [[@LINE-1]]:16: error: cannot coerce from fruits.secret(fruits.citrus) to fruits.secret(fruits.citrus)
% CHECK: [[@LINE-2]]:16: because: fruits.secret(fruits.citrus) is not a discriminated union type whose definition is visible
