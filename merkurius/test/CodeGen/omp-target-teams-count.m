% A number of teams that is not positive stops the program before the
% region starts, as libomptarget would take it for its default.
%
% RUN: merkc -fopenmp %libomp-flags %s -o %t && not %libomp-env %t 2>&1 | FileCheck %s --match-full-lines --strict-whitespace
% REQUIRES: libomp

:- module omp_target_teams_count.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- pragma no_inline(none/1).
:- pred none(int::out) is det.
none(0).

main(!IO) :-
    none(N),
    %$omp target teams num_teams(N)
    io.write_string("in a team\n", !IO),
    io.write_int(N, !IO),
    io.nl(!IO).
% CHECK:Uncaught Mercury exception:
% CHECK-NEXT:Software Error: target teams: the number of teams, 0, is not from 1 to 2147483647
% CHECK-NOT:{{.}}
