% MANUAL: Syntax and semantics of type specialization pragmas
% A substitution gives a variable one type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_two_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred e(T::in, U::in) is det.
:- pragma type_spec(e/2, (T = int, T = string)). % ERROR
e(_, _).
main(!IO) :- io.write_string("ok\n", !IO).
