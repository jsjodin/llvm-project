% MANUAL: Goals
% require_complete_switch of a switch without an arm for each
% constructor is an error.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_incomplete_switch.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- type c ---> r ; g ; b.
:- pred colour(c::in, int::out) is semidet.
colour(C, N) :-
    require_complete_switch [C] % ERROR
    ( C = r, N = 1
    ; C = g, N = 2
    ).
main(!IO) :- io.write_string("ok\n", !IO).
