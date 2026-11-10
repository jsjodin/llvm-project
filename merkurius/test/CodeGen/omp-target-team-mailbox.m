% Team mailboxes (docs/INPROGRESS.md, Processes as teams, step 2): the teams
% of a target teams region pass a token around a ring, each through the next
% team's mailbox; then each sends its number to team 0 many times, more than
% a device's ring holds, and a yes in a mailbox of another type. On a device
% the mailboxes are rings in its memory; on the host and without OpenMP,
% ordinary mailboxes. Outside a teams region the only team is 0. A plain
% target region in a module with team mailboxes says that it has no teams.
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-merk %s | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o /dev/null
% RUN: merkc -O0 -fopenmp %libomp-flags %s -o %t0 && %libomp-env %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t2 && %libomp-env %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_team_mailbox.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module bool, int, list, process.

main(!IO) :-
    new_mailbox(Ring, !IO),
    new_mailbox(Lost, !IO),
    new_mailbox(Gathered, !IO),
    %$omp target teams num_teams(4)
    ring(Ring, Lost, !IO),
    %$omp target teams num_teams(4)
    gather(Gathered, !IO),
    collect(Ring, [], Hops, !IO),
    ( if Hops = [0] then
        io.write_string("the token came back after a hop for each team\n",
            !IO)
    else
        io.write_string("the token's hops less the teams: ", !IO),
        io.print_line(Hops, !IO)
    ),
    collect(Lost, [], LostTeams, !IO),
    ( if LostTeams = [] then
        true
    else
        io.write_string("teams that got no token: ", !IO),
        list.sort(LostTeams, Sorted),
        io.print_line(Sorted, !IO)
    ),
    collect(Gathered, [], Reversed, !IO),
    ( if Reversed = [0, 0, messages] then
        io.write_string("team 0 received 300 numbers per team ", !IO),
        io.write_string("with the right sum and a yes from each\n", !IO)
    else
        io.write_string("team 0 received per team, and the errors: ", !IO),
        list.reverse(Reversed, Received),
        io.print_line(Received, !IO)
    ),
    % Outside a teams region, team 0's mailbox, the same each time.
    team_mailbox(0, Mine, !IO),
    send(Mine, 7, !IO),
    team_mailbox(0, Again, !IO),
    receive(Again, any, after(0), Outside : received(int), !IO),
    io.write_string("outside: ", !IO),
    io.print_line(Outside, !IO),
    X = 20,
    %$omp target
    Y = X + 1,
    io.write_int(Y, !IO),
    io.nl(!IO).

:- func messages = int.

messages = 300.

:- pred ring(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.

ring(Reports, Lost, !IO) :-
    team_number(Me, !IO),
    number_of_teams(All, !IO),
    team_mailbox((Me + 1) mod All, Next, !IO),
    team_mailbox(Me, Mine, !IO),
    ( if Me = 0 then
        send(Next, 1, !IO)
    else
        true
    ),
    receive(Mine, any, after(10000), Token, !IO),
    (
        Token = message(Hops),
        ( if Me = 0 then
            send(Reports, Hops - All, !IO)
        else
            send(Next, Hops + 1, !IO)
        )
    ;
        Token = timeout,
        send(Lost, Me, !IO)
    ).

:- pred gather(mailbox(int)::in, io::di, io::uo) is det.

gather(Reports, !IO) :-
    team_number(Me, !IO),
    number_of_teams(All, !IO),
    team_mailbox(0, First, !IO),
    team_mailbox(0, Answers, !IO),
    send_many(First, Me, messages, !IO),
    send(Answers, yes, !IO),
    ( if Me = 0 then
        gather_ints(First, All * messages, 0, Sum, 0, Count, !IO),
        gather_bools(Answers, All, 0, Yes, !IO),
        send(Reports, Count / All, !IO),
        send(Reports, Sum - messages * All * (All - 1) / 2, !IO),
        send(Reports, Yes - All, !IO)
    else
        true
    ).

:- pred send_many(mailbox(int)::in, int::in, int::in, io::di, io::uo) is det.

send_many(Box, Message, Left, !IO) :-
    ( if Left > 0 then
        send(Box, Message, !IO),
        send_many(Box, Message, Left - 1, !IO)
    else
        true
    ).

    % Receives up to Left messages, adding them up and counting them.
:- pred gather_ints(mailbox(int)::in, int::in, int::in, int::out, int::in,
    int::out, io::di, io::uo) is det.

gather_ints(Box, Left, !Sum, !Count, !IO) :-
    ( if Left > 0 then
        receive(Box, any, after(10000), Got, !IO),
        (
            Got = message(Number),
            !:Sum = !.Sum + Number,
            !:Count = !.Count + 1,
            gather_ints(Box, Left - 1, !Sum, !Count, !IO)
        ;
            Got = timeout
        )
    else
        true
    ).

:- pred gather_bools(mailbox(bool)::in, int::in, int::in, int::out,
    io::di, io::uo) is det.

gather_bools(Box, Left, !Yes, !IO) :-
    ( if Left > 0 then
        receive(Box, any, after(10000), Got, !IO),
        (
            Got = message(Answer),
            ( if Answer = yes then
                !:Yes = !.Yes + 1
            else
                true
            ),
            gather_bools(Box, Left - 1, !Yes, !IO)
        ;
            Got = timeout
        )
    else
        true
    ).

:- pred collect(mailbox(int)::in, list(int)::in, list(int)::out,
    io::di, io::uo) is det.

collect(Box, !Got, !IO) :-
    receive(Box, any, after(0), Received, !IO),
    (
        Received = message(Number),
        collect(Box, [Number | !.Got], !:Got, !IO)
    ;
        Received = timeout
    ).

% MERK-LABEL: merk.proc @omp_target_team_mailbox.main_2_0()
% MERK: %[[BEGIN:.*]]:3 = merk.foreign_call @merk_teams_begin() : () -> (i64, i64, i64)
% MERK: omp.target kernel_type(generic)
% MERK: omp.teams num_teams(
% MERK-NEXT: merk.foreign_call @merk_device_region_enter(%{{.*}}) : (i64) -> ()
% MERK-NEXT: %[[TEAMS:.*]] = merk.foreign_call @omp_get_num_teams() : () -> i32
% MERK-NEXT: merk.foreign_call @merk_device_teams_enter(%{{.*}}, %{{.*}}, %{{.*}}, %[[TEAMS]]) : (i64, i64, i64, i32) -> ()
% MERK-NEXT: merk.call @omp_target_team_mailbox.ring_4_0
% MERK-NEXT: merk.foreign_call @merk_device_teams_leave() : () -> ()
% MERK-NEXT: omp.terminator
% MERK: merk.foreign_call @merk_region_exit(%{{.*}}) : (i64) -> ()
% MERK-NEXT: merk.foreign_call @merk_teams_end(%[[BEGIN]]#0) : (i64) -> ()
% MERK-NOT: merk_device_no_teams
% MERK: merk.call @omp_target_team_mailbox.gather_3_0
% MERK-LABEL: merk.proc private @omp_target_team_mailbox.ring_4_0(
% MERK: merk.call @process.team_mailbox_4_0(

% DEVICE-LABEL: merk.proc @omp_target_team_mailbox.main_2_0()
% DEVICE: omp.teams num_teams(
% DEVICE-NEXT: merk.foreign_call @merk_device_region_enter(
% DEVICE-NEXT: merk.foreign_call @omp_get_num_teams()
% DEVICE-NEXT: merk.foreign_call @merk_device_teams_enter(
% DEVICE-NEXT: merk.foreign_call @merk_device_teams_wait_all() : () -> ()
% DEVICE-NOT: merk_device_no_teams
% DEVICE: merk.call @omp_target_team_mailbox.gather_3_0
% DEVICE-NOT: merk_device_no_teams
% DEVICE: omp.target kernel_type(generic)
% DEVICE-NEXT: merk.foreign_call @merk_device_no_teams() : () -> ()
% DEVICE-LABEL: merk.proc private @omp_target_team_mailbox.ring_4_0(
% DEVICE: %[[NEXT:.*]] = merk.foreign_call @merk_device_team_mailbox(%{{.*}}, %{{.*}}) : (i64, i64) -> !llvm.ptr
% DEVICE-NEXT: builtin.unrealized_conversion_cast %[[NEXT]] : !llvm.ptr to !merk.term<"process.mailbox(int)">
% DEVICE-NOT: process.team_mailbox

% OUTPUT:the token came back after a hop for each team
% OUTPUT-NEXT:team 0 received 300 numbers per team with the right sum and a yes from each
% OUTPUT-NEXT:outside: message(7)
% OUTPUT-NEXT:21
% OUTPUT-NOT:{{.}}
