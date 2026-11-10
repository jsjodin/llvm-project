% Arrays in team mailboxes (docs/INPROGRESS.md, Processes as teams, step 3).
% Team 0 makes an array and sends it to every team, and an array that the
% host gave the region to the last team. Each team sums its slice of the
% array for team 0, changes a copy of it, and sends team 0 a piece of its
% own and many small arrays, more than a device's ring holds. On a device an
% array goes by its pointer, since every team can address it; neither team
% can change it in place, as a send makes the sender's array shared and a
% receive gives a shared one, so a team that changes its copy does not
% change the others'. On the host and without OpenMP, team mailboxes copy
% as other mailboxes do.
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-merk %s | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o /dev/null
% RUN: merkc -O0 -fopenmp %libomp-flags %s -o %t0 && %libomp-env %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t2 && %libomp-env %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_team_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, process.

main(!IO) :-
    new_mailbox(Reports, !IO),
    new_mailbox(Copies, !IO),
    Given = array.from_list([1, 2, 3, 4]) : array(int),
    %$omp target teams num_teams(4)
    share(Given, Reports, Copies, !IO),
    collect(Reports, [], Reversed, !IO),
    list.reverse(Reversed, Errors),
    ( if Errors = [0, 0, 0, 0, 0, 0] then
        io.write_string("team 0 got every slice's sum, piece and small array, ",
            !IO),
        io.write_string("and its array was unchanged\n", !IO)
    else
        io.write_string("team 0's errors: ", !IO),
        io.print_line(Errors, !IO)
    ),
    collect(Copies, [], Changed, !IO),
    ( if Changed = [] then
        io.write_string("each team changed only its own copy\n", !IO)
    else
        io.write_string("teams whose copy was wrong: ", !IO),
        list.sort(Changed, Sorted),
        io.print_line(Sorted, !IO)
    ).

:- func size = int.

size = 1000.

:- func small_arrays = int.

small_arrays = 100.

:- func given_sum = int.

given_sum = 10.

:- pred share(array(int)::in, mailbox(int)::in, mailbox(int)::in,
    io::di, io::uo) is det.

share(Given, Reports, Copies, !IO) :-
    team_number(Me, !IO),
    number_of_teams(All, !IO),
    team_mailbox(Me, Mine : mailbox(array(int)), !IO),
    team_mailbox(0, Sums : mailbox(int), !IO),
    team_mailbox(0, Pieces : mailbox(array(int)), !IO),
    team_mailbox(0, Smalls : mailbox(array(float)), !IO),
    ( if Me = 0 then
        count_up(0, size, array.init(size, 0), Made),
        send_to_teams(Made, 0, All, !IO),
        team_mailbox(All - 1, Last, !IO),
        send(Last, Given, !IO)
    else
        true
    ),
    receive(Mine, any, after(10000), Got, !IO),
    (
        Got = message(Whole),
        Low = Me * size / All,
        High = (Me + 1) * size / All,
        send(Sums, total(Whole, Low, High, 0), !IO),
        check_copy(Whole, Me, Copies, !IO),
        send(Pieces, doubled(Whole, Low, High), !IO),
        send_smalls(Smalls, float(Me), small_arrays, !IO),
        ( if Me = All - 1 then
            receive(Mine, any, after(10000), GotGiven, !IO),
            (
                GotGiven = message(TheGiven),
                send(Sums, total(TheGiven, 0, array.size(TheGiven), 0), !IO)
            ;
                GotGiven = timeout
            )
        else
            true
        ),
        ( if Me = 0 then
            check(Whole, All, Mine, Sums, Smalls, Reports, !IO)
        else
            true
        )
    ;
        Got = timeout
    ).

:- pred count_up(int::in, int::in,
    array(int)::array_di, array(int)::array_uo) is det.

count_up(I, N, !A) :-
    ( if I < N then
        array.set(I, I, !A),
        count_up(I + 1, N, !A)
    else
        true
    ).

:- pred send_to_teams(array(int)::in, int::in, int::in, io::di, io::uo)
    is det.

send_to_teams(Array, Team, All, !IO) :-
    ( if Team < All then
        team_mailbox(Team, Box, !IO),
        send(Box, Array, !IO),
        send_to_teams(Array, Team + 1, All, !IO)
    else
        true
    ).

    % The sum of the elements from Low up to High.
:- func total(array(int), int, int, int) = int.

total(A, I, High, Sum) =
    ( if I >= High then
        Sum
    else
        total(A, I + 1, High, Sum + A ^ elem(I))
    ).

    % The elements from Low up to High, doubled, in an array of their own.
:- func doubled(array(int), int, int) = array(int).

doubled(Whole, Low, High) = Piece :-
    double_into(Whole, Low, 0, High - Low, array.init(High - Low, 0), Piece).

:- pred double_into(array(int)::in, int::in, int::in, int::in,
    array(int)::array_di, array(int)::array_uo) is det.

double_into(Whole, Low, I, N, !Piece) :-
    ( if I < N then
        array.set(I, 2 * Whole ^ elem(Low + I), !Piece),
        double_into(Whole, Low, I + 1, N, !Piece)
    else
        true
    ).

    % A copy of the array with the team's element changed: if the copy were
    % the array itself, another team's copy, or the array, would show it.
:- pred check_copy(array(int)::in, int::in, mailbox(int)::in,
    io::di, io::uo) is det.

check_copy(Whole, Me, Copies, !IO) :-
    Copy0 = array.copy(Whole),
    array.set(Me, -1, Copy0, Copy),
    Size = array.size(Whole),
    ( if
        total(Copy, 0, Size, 0) - total(Whole, 0, Size, 0) = -1 - Me,
        Whole ^ elem(Me) = Me
    then
        true
    else
        send(Copies, Me, !IO)
    ).

:- pred send_smalls(mailbox(array(float))::in, float::in, int::in,
    io::di, io::uo) is det.

send_smalls(Smalls, Value, Left, !IO) :-
    ( if Left > 0 then
        send(Smalls, array.init(2, Value), !IO),
        send_smalls(Smalls, Value, Left - 1, !IO)
    else
        true
    ).

    % Team 0's checks, each 0 when right: the slices' sums and the given
    % array's, the pieces' sizes and sums, the small arrays' number and sum,
    % and the array it made, after every team has used it.
:- pred check(array(int)::in, int::in, mailbox(array(int))::in,
    mailbox(int)::in, mailbox(array(float))::in, mailbox(int)::in,
    io::di, io::uo) is det.

check(Whole, All, Pieces, Sums, Smalls, Reports, !IO) :-
    Expected = size * (size - 1) / 2,
    add_ints(Sums, All + 1, 0, Sum, !IO),
    add_pieces(Pieces, All, 0, Elements, 0, Doubled, !IO),
    add_smalls(Smalls, All * small_arrays, 0, Count, 0.0, SmallSum, !IO),
    send(Reports, Sum - Expected - given_sum, !IO),
    send(Reports, Elements - size, !IO),
    send(Reports, Doubled - 2 * Expected, !IO),
    send(Reports, Count - All * small_arrays, !IO),
    ( if SmallSum = float(small_arrays * All * (All - 1)) then
        send(Reports, 0, !IO)
    else
        send(Reports, 1, !IO)
    ),
    send(Reports, total(Whole, 0, size, 0) - Expected, !IO).

:- pred add_ints(mailbox(int)::in, int::in, int::in, int::out,
    io::di, io::uo) is det.

add_ints(Box, Left, !Sum, !IO) :-
    ( if Left > 0 then
        receive(Box, any, after(10000), Got, !IO),
        (
            Got = message(Number),
            !:Sum = !.Sum + Number,
            add_ints(Box, Left - 1, !Sum, !IO)
        ;
            Got = timeout
        )
    else
        true
    ).

:- pred add_pieces(mailbox(array(int))::in, int::in, int::in, int::out,
    int::in, int::out, io::di, io::uo) is det.

add_pieces(Box, Left, !Elements, !Sum, !IO) :-
    ( if Left > 0 then
        receive(Box, any, after(10000), Got, !IO),
        (
            Got = message(Piece),
            Size = array.size(Piece),
            !:Elements = !.Elements + Size,
            !:Sum = !.Sum + total(Piece, 0, Size, 0),
            add_pieces(Box, Left - 1, !Elements, !Sum, !IO)
        ;
            Got = timeout
        )
    else
        true
    ).

:- pred add_smalls(mailbox(array(float))::in, int::in, int::in, int::out,
    float::in, float::out, io::di, io::uo) is det.

add_smalls(Box, Left, !Count, !Sum, !IO) :-
    ( if Left > 0 then
        receive(Box, any, after(10000), Got, !IO),
        (
            Got = message(Small),
            !:Count = !.Count + 1,
            !:Sum = !.Sum + Small ^ elem(0) + Small ^ elem(1),
            add_smalls(Box, Left - 1, !Count, !Sum, !IO)
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

% MERK-LABEL: merk.proc @omp_target_team_arrays.main_2_0()
% MERK: omp.teams num_teams(
% MERK: merk.call @omp_target_team_arrays.share_5_0
% MERK-LABEL: merk.proc private @omp_target_team_arrays.share_5_0(
% MERK: merk.call @process.team_mailbox_4_0(

% DEVICE-LABEL: merk.proc private @omp_target_team_arrays.share_5_0(
% DEVICE: %[[MINE:.*]] = merk.foreign_call @merk_device_team_mailbox(%{{.*}}, %{{.*}}) : (i64, i64) -> !llvm.ptr
% DEVICE-NEXT: builtin.unrealized_conversion_cast %[[MINE]] : !llvm.ptr to !merk.term<"process.mailbox(array.array(int))">
% DEVICE: merk.foreign_call @merk_device_receive_array(
% DEVICE: merk.foreign_call @merk_device_send_array(
% DEVICE-NOT: process.team_mailbox

% OUTPUT:team 0 got every slice's sum, piece and small array, and its array was unchanged
% OUTPUT-NEXT:each team changed only its own copy
% OUTPUT-NOT:{{.}}
