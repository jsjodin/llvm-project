% A type parameter in a tuple is not one of the places where a coerce must
% keep it the same (reference manual, "Type conversions"), so this coerce
% is type-correct, as mmc agrees; mmc's mode analysis of it crashes
% ("missing constructor for result type"), so it is not compared (see the
% register).
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s

:- module tuple_parameter.
:- interface.
:- pred main(int::out) is det.
:- implementation.

main(0).
:- type fruit ---> apple ; lemon ; orange.
:- type citrus =< fruit ---> lemon ; orange.
:- type in_tuple(T) ---> in_tuple({T, int}).
:- func of_tuple(in_tuple(citrus)) = in_tuple(fruit).
of_tuple(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func tuple_parameter.of_tuple/1: func tuple_parameter.of_tuple(tuple_parameter.in_tuple(tuple_parameter.citrus)) = tuple_parameter.in_tuple(tuple_parameter.fruit)
