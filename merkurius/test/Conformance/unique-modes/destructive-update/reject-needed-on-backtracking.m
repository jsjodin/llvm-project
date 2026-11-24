% MANUAL: Destructive update
% A value that may be needed on backtracking is not unique, so the state
% of the world may not be destroyed where a disjunction could backtrack
% into it. A predicate with I/O states may not be multi, either.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:19:{{[0-9]+}}: error:

:- module reject_needed_on_backtracking.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- pred writes(io::di, io::uo) is multi.             % ERROR
writes(!IO) :-
    ( X = 1 ; X = 2 ),
    io.write_int(X, !IO).                           % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
