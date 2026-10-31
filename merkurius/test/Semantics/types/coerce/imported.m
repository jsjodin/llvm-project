% Coerces of imported types: an imported subtype is coerced like any
% other, but an abstractly exported subtype is not a discriminated union
% type here (reference manual, "Type conversions").
%
% RUN: not merkc -fc1 -fdebug-dump-types -I %S/Inputs %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines --include %S/Inputs %s %}

:- module imported.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module fruits, list.

main(0).
:- func of_citrus(list(citrus)) = list(fruit).
of_citrus(X) = coerce(X).
:- func of_pome(pome) = fruit.
of_pome(X) = coerce(X).
% CHECK: [[@LINE-1]]:14: error: cannot coerce from fruits.pome to fruits.fruit
% CHECK: [[@LINE-2]]:14: because: fruits.pome is not a discriminated union type whose definition is visible
