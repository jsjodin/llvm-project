% MANUAL: Exception handling
% CATCHANYVAR has type some [T] T, which is not int.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:19:{{[0-9]+}}: error:

:- module reject_catch_any_type.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    ( try [] throw(3)
    then
        io.write_string("then\n", !IO)
    catch_any E ->
        io.write_int(E, !IO) % ERROR
    ).
