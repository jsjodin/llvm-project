% MANUAL: Solver type definitions
% A solver type's comparison predicate must take a comparison_result and
% two values of the type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_comparison_type.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- solver type t where representation is int, comparison is cmp. % ERROR
:- pred cmp(comparison_result::uo, int::in, int::in) is det.
cmp(R, A, B) :- compare(R, A, B).

main(!IO) :- io.write_string("ok\n", !IO).
