% A procedure on a device that builds a list after its recursive call, so
% that last call modulo constructor makes a variant of it, and whose body
% has a foreign call: here team_mailbox, which the device's compile makes a
% call of merk_device_team_mailbox. The variant is a copy with a foreign
% call of its own, whose word inputs lowering must know too; it crashed
% the device's compile when they were found before that pass.
%
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t2 && %libomp-env %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=SEQUENTIAL --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_lcmc_foreign.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process.

main(!IO) :-
    new_mailbox(Sums, !IO),
    %$omp target teams num_teams(4)
    to_every_team(Sums, !IO),
    receive(Sums, any, after(10000), Got, !IO),
    io.print_line(Got, !IO).

    % Each team sends its number to every team; team 0 sends the sum of what
    % it received to Sums.
:- pred to_every_team(mailbox(int)::in, io::di, io::uo) is det.

to_every_team(Sums, !IO) :-
    team_number(Me, !IO),
    number_of_teams(All, !IO),
    mailboxes(0, All, Boxes, !IO),
    send_all(Boxes, Me, !IO),
    ( if Me = 0 then
        team_mailbox(0, Mine, !IO),
        add_received(Mine, All, 0, Sum, !IO),
        send(Sums, Sum, !IO)
    else
        true
    ).

:- pred mailboxes(int::in, int::in, list(mailbox(int))::out, io::di, io::uo)
    is det.

mailboxes(From, All, Boxes, !IO) :-
    ( if From < All then
        team_mailbox(From, Box, !IO),
        mailboxes(From + 1, All, Rest, !IO),
        Boxes = [Box | Rest]
    else
        Boxes = []
    ).

:- pred send_all(list(mailbox(int))::in, int::in, io::di, io::uo) is det.

send_all([], _, !IO).
send_all([Box | Boxes], Message, !IO) :-
    send(Box, Message, !IO),
    send_all(Boxes, Message, !IO).

:- pred add_received(mailbox(int)::in, int::in, int::in, int::out,
    io::di, io::uo) is det.

add_received(Box, Left, !Sum, !IO) :-
    ( if Left > 0 then
        receive(Box, any, after(10000), Got, !IO),
        (
            Got = message(Number),
            add_received(Box, Left - 1, !.Sum + Number, !:Sum, !IO)
        ;
            Got = timeout
        )
    else
        true
    ).

% DEVICE-LABEL: define internal ptr @omp_target_lcmc_foreign.mailboxes_5_0(
% DEVICE: call ptr @merk_device_team_mailbox(i64 %0, i64 {{-?[0-9]+}})
% The variant, which fills the list's cell, given its hole.
% DEVICE-LABEL: define internal void @omp_target_lcmc_foreign.mailboxes_5_0.lcmc(
% DEVICE: call ptr @merk_device_team_mailbox(i64 %0, i64 {{-?[0-9]+}})

% OUTPUT:message(6)
% OUTPUT-NOT:{{.}}
% SEQUENTIAL:message(0)
% SEQUENTIAL-NOT:{{.}}
