% The teams of a target teams region whose code uses team mailboxes wait until
% all have started (docs/INPROGRESS.md, Processes as teams, step 4), as a team
% that is not running would deadlock the teams waiting for its messages. Code
% uses them if it calls team_mailbox, or calls a procedure that does, at any
% depth. A teams region whose code does not use them does not wait, so its
% teams may run a few at a time, more than fit on the device at once.
%
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-merk %s | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=HOST
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t2 && %libomp-env %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=SEQUENTIAL --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_teams_wait.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, process.

main(!IO) :-
    new_mailbox(Counts, !IO),
    %$omp target teams num_teams(3)
    count_in(Counts, !IO),
    %$omp target teams num_teams(3)
    report_to_first(Counts, !IO),
    add_up(Counts, 0, Total, !IO),
    io.write_string("counted ", !IO),
    io.write_int(Total, !IO),
    io.nl(!IO).

    % No team mailboxes: each team sends 1 to the host's mailbox.
:- pred count_in(mailbox(int)::in, io::di, io::uo) is det.

count_in(Counts, !IO) :-
    send(Counts, 1, !IO).

    % Team mailboxes two calls down: each team sends 10 to team 0, which
    % sends their sum to the host's mailbox.
:- pred report_to_first(mailbox(int)::in, io::di, io::uo) is det.

report_to_first(Counts, !IO) :-
    team_number(Me, !IO),
    number_of_teams(All, !IO),
    send_ten(!IO),
    ( if Me = 0 then
        team_mailbox(0, First, !IO),
        add_received(First, All, 0, Sum, !IO),
        send(Counts, Sum, !IO)
    else
        true
    ).

:- pred send_ten(io::di, io::uo) is det.

send_ten(!IO) :-
    first_mailbox(First, !IO),
    send(First, 10, !IO).

:- pred first_mailbox(mailbox(int)::out, io::di, io::uo) is det.

first_mailbox(First, !IO) :-
    team_mailbox(0, First, !IO).

:- pred add_received(mailbox(int)::in, int::in, int::in, int::out,
    io::di, io::uo) is det.

add_received(Box, Left, !Sum, !IO) :-
    ( if Left > 0 then
        receive(Box, any, after(10000), Got, !IO),
        (
            Got = message(Number),
            !:Sum = !.Sum + Number,
            add_received(Box, Left - 1, !Sum, !IO)
        ;
            Got = timeout
        )
    else
        true
    ).

:- pred add_up(mailbox(int)::in, int::in, int::out, io::di, io::uo) is det.

add_up(Box, !Total, !IO) :-
    receive(Box, any, after(0), Got, !IO),
    (
        Got = message(Number),
        add_up(Box, !.Total + Number, !:Total, !IO)
    ;
        Got = timeout
    ).

% DEVICE-LABEL: merk.proc @omp_target_teams_wait.main_2_0()
% DEVICE: merk.foreign_call @merk_device_teams_enter(
% DEVICE-NEXT: merk.call @omp_target_teams_wait.count_in_3_0
% DEVICE: merk.foreign_call @merk_device_teams_enter(
% DEVICE-NEXT: merk.foreign_call @merk_device_teams_wait_all() : () -> ()
% DEVICE-NEXT: merk.call @omp_target_teams_wait.report_to_first_3_0

% The host's teams are threads, which all run.
% HOST-NOT: merk_device_teams_wait_all

% With OpenMP, 3 teams count in 1 each and send 10 each; without, 1 team.
% OUTPUT:counted 33
% OUTPUT-NOT:{{.}}
% SEQUENTIAL:counted 11
% SEQUENTIAL-NOT:{{.}}
