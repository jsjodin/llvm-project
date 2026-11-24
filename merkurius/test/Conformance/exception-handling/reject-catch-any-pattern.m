% MANUAL: Exception handling
% CATCHANYVAR must be a single variable.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:21:{{[0-9]+}}: error:

:- module reject_catch_any_pattern.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

:- type t
    --->    f(int).

main(!IO) :-
    ( try [] throw(f(1))
    then
        io.write_string("then\n", !IO)
    catch_any f(E) -> % ERROR
        io.write_int(E, !IO)
    ).
