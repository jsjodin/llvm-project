% MANUAL: Discriminated unions
% There must not be more than one constructor with the same name, arity and
% result type in a module.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_duplicate_constructor.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t
    --->    a
    ;       b
    ;       a.                                      % ERROR

main(!IO) :-
    io.write_line(b, !IO).
