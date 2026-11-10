% Team mailboxes on a device hold one-word messages and arrays of them so far
% (docs/INPROGRESS.md, Processes as teams, steps 2 and 3). A team mailbox
% for another type, such as strings, terms with arguments or arrays of
% strings, is an error at team_mailbox in the device's compile. The host's
% compile calls the library's team_mailbox, whose mailboxes hold any type.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: merkc -fc1 -fopenmp -emit-merk %s -o /dev/null
% REQUIRES: amdgpu-registered-target

:- module omp_target_team_mailbox_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, process.

:- type point
    --->    point(int, int).

main(!IO) :-
    strings(!IO),
    words(!IO),
    points(!IO).

:- pred strings(io::di, io::uo) is det.
strings(!IO) :-
    %$omp target teams num_teams(2)
    team_mailbox(0, _ : mailbox(string), !IO).
% CHECK: omp-target-team-mailbox-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for team mailboxes for messages of type string on a device

:- pred words(io::di, io::uo) is det.
words(!IO) :-
    %$omp target teams num_teams(2)
    team_mailbox(0, _ : mailbox(array(string)), !IO).
% CHECK: omp-target-team-mailbox-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for team mailboxes for messages of type array.array(string) on a device

:- pred points(io::di, io::uo) is det.
points(!IO) :-
    %$omp target teams num_teams(2)
    team_mailbox(0, _ : mailbox(point), !IO).
% CHECK: omp-target-team-mailbox-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for team mailboxes for messages of type omp_target_team_mailbox_errors.point on a device
