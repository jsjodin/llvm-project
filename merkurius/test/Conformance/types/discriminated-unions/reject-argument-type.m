% MANUAL: Discriminated unions
% A constructor's arguments must have the types its definition gives
% them.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:18:{{[0-9]+}}: error:

:- module reject_argument_type.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- type box
    --->    box(int).

main(!IO) :-
    B = box("one"),                                 % ERROR
    io.write_line(B, !IO).
