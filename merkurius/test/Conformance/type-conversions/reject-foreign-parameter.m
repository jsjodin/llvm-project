% MANUAL: Type conversions
% A parameter must be the same in both types if the base type's definition
% has it in a foreign type: holds(T)'s argument is a handle(T), a C
% pointer.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:36:{{[0-9]+}}: error:

:- module reject_foreign_parameter.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- type fruit
    --->    apple
    ;       lemon
    ;       orange.

:- type citrus =< fruit
    --->    lemon
    ;       orange.

:- type handle(T).
:- pragma foreign_type("C", handle(T), "void *").

:- func null_handle = handle(T).
:- pragma foreign_proc("C", null_handle = (H::out),
    [will_not_call_mercury, promise_pure], "H = NULL;").

:- type holds(T)
    --->    holds(handle(T)).

main(!IO) :-
    X = holds(null_handle) : holds(citrus),
    Y = coerce(X) : holds(fruit),                       % ERROR
    io.write_string("converted\n", !IO),
    ( if Y = Y then true else true ).
