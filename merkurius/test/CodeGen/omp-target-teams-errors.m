% What a target teams region does not take yet, or at all.
%
% RUN: not merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s 2>&1 | FileCheck %s --implicit-check-not=error:
% REQUIRES: amdgpu-registered-target

:- module omp_target_teams_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int.

main(!IO) :-
    a(!IO),
    b(1.0, !IO),
    c(!IO),
    d(1, !IO),
    e(!IO),
    f(!IO),
    g(1, Y, !IO),
    io.write_int(Y, !IO).

:- pred a(io::di, io::uo) is det.
a(!IO) :-
    %$omp target teams num_teams(Unbound)
    io.write_string("a\n", !IO).
% CHECK: omp-target-teams-errors.m:[[@LINE-2]]:5: error: the number of teams Unbound is not bound before the directive

:- pred b(float::in, io::di, io::uo) is det.
b(F, !IO) :-
    %$omp target teams num_teams(F)
    io.write_string("b\n", !IO).
% CHECK: omp-target-teams-errors.m:[[@LINE-2]]:5: error: the number of teams F must be an integer

:- pred c(io::di, io::uo) is det.
c(!IO) :-
    %$omp target teams num_teams(0)
    io.write_string("c\n", !IO).
% CHECK: omp-target-teams-errors.m:[[@LINE-2]]:5: error: the number of teams must be from 1 to 2147483647

:- pred d(int::in, io::di, io::uo) is det.
d(X, !IO) :-
    %$omp target teams num_teams(X + 1)
    io.write_int(X, !IO).
% CHECK: omp-target-teams-errors.m:[[@LINE-2]]:5: error: not yet implemented: {{.*}}a number of teams that is not an integer or a variable

:- pred e(io::di, io::uo) is det.
e(!IO) :-
    %$omp target teams num_teams(1) num_teams(2)
    io.write_string("e\n", !IO).
% CHECK: omp-target-teams-errors.m:[[@LINE-2]]:5: error: {{.*}}

:- pred f(io::di, io::uo) is det.
f(!IO) :-
    %$omp target teams thread_limit(64)
    io.write_string("f\n", !IO).
% CHECK: omp-target-teams-errors.m:[[@LINE-2]]:5: error: not yet implemented: {{.*}}the thread_limit clause of the target directive

:- pred g(int::in, int::out, io::di, io::uo) is det.
g(X, Y, !IO) :-
    %$omp target teams
    Y = X + 1.
% CHECK: omp-target-teams-errors.m:[[@LINE-2]]:5: error: a target teams region must not bind Y, which is used after it, since every team would bind it
