% process.team_barrier (docs/INPROGRESS.md, Library code on the device, step
% 5): in each round, every team of a target teams region sends the round's
% number to team 0, and all wait at the barrier; then team 0 polls, without
% waiting, for every team's message, and all wait again before the next
% round. On a device the barrier is a counter in the region's team table,
% whose teams wait for each other to start; on the host's teams, the same
% with a wait that blocks. Outside a teams region, and without OpenMP, there
% is one team, and the barrier returns at once.
%
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-merk %s | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
% RUN: merkc -O0 -fopenmp %libomp-flags %s -o %t0 && %libomp-env %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t2 && %libomp-env %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=SEQUENTIAL --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_team_barrier.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, process.

main(!IO) :-
    team_barrier(!IO),
    io.write_string("outside\n", !IO),
    %$omp target
    alone(!IO),
    new_mailbox(Reports, !IO),
    %$omp target teams num_teams(4)
    rounds(Reports, 0, !IO),
    receive(Reports, any, after(10000), Got, !IO),
    io.write_string("wrong: ", !IO),
    io.print_line(Got : received(int), !IO).

:- pred alone(io::di, io::uo) is det.

alone(!IO) :-
    team_barrier(!IO),
    io.write_string("alone\n", !IO).

:- func rounds = int.

rounds = 50.

:- pred rounds(mailbox(int)::in, int::in, io::di, io::uo) is det.

rounds(Reports, Wrong, !IO) :-
    rounds_from(0, Wrong, Total, !IO),
    team_number(Me, !IO),
    ( if Me = 0 then
        send(Reports, Total, !IO)
    else
        true
    ).

:- pred rounds_from(int::in, int::in, int::out, io::di, io::uo) is det.

rounds_from(R, !Wrong, !IO) :-
    ( if R < rounds then
        team_number(Me, !IO),
        number_of_teams(All, !IO),
        team_mailbox(0, First, !IO),
        send(First, R, !IO),
        team_barrier(!IO),
        ( if Me = 0 then
            poll(First, R, All, !Wrong, !IO)
        else
            true
        ),
        team_barrier(!IO),
        rounds_from(R + 1, !Wrong, !IO)
    else
        true
    ).

    % Takes Left messages that have arrived already, counting those that are
    % missing or not R.
:- pred poll(mailbox(int)::in, int::in, int::in, int::in, int::out,
    io::di, io::uo) is det.

poll(Box, R, Left, !Wrong, !IO) :-
    ( if Left > 0 then
        receive(Box, any, after(0), Got, !IO),
        ( if Got = message(R) then
            true
        else
            !:Wrong = !.Wrong + 1
        ),
        poll(Box, R, Left - 1, !Wrong, !IO)
    else
        true
    ).

% The teams region waits for its teams to start; the other region says that
% it has none.
% DEVICE: merk.foreign_call @merk_device_no_teams()
% DEVICE: merk.foreign_call @merk_device_teams_enter(
% DEVICE-NEXT: merk.foreign_call @merk_device_teams_wait_all()
% DEVICE: merk.proc private @process.team_barrier_2_0(

% LLVM: call void @merk_device_team_barrier()

% OUTPUT:outside
% OUTPUT-NEXT:alone
% OUTPUT-NEXT:wrong: message(0)
% OUTPUT-NOT:{{.}}

% SEQUENTIAL:outside
% SEQUENTIAL-NEXT:alone
% SEQUENTIAL-NEXT:wrong: message(0)
% SEQUENTIAL-NOT:{{.}}
