% MANUAL: Predicate and function mode declarations
% A predicate whose type and mode declarations are combined may not have
% other mode declarations.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_combined_and_separate.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- pred p(int::in, int::out) is det.
:- mode p(out, in) is det.                          % ERROR
p(X, X).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
