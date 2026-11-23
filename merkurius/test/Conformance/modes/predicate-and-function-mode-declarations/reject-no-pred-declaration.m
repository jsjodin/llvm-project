% MANUAL: Predicate and function mode declarations
% A mode declaration needs the predicate's type declaration.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_no_pred_declaration.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- mode p(in) is det.                               % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
