% MANUAL: Signed integer types
% An integer literal must fit its type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_literal_too_large.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int8.

main(!IO) :-
    X = 128i8,                                      % ERROR
    io.write_line(X, !IO).
