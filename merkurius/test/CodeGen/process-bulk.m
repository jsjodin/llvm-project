% Large fills and copies of a GPU process with worker teams, which the
% teams do for it (runtime/device/device.cpp, Bulk; docs/INPROGRESS.md,
% Messages between GPU processes on one GPU, step 3): array.init, fill,
% copy and resize of arrays of 1, 2, 4 and 8 bytes an element, of sizes that
% end within a word, a resize's fill that starts within one, array.init and
% resize of an array of flat records of three words, which a team's threads
% do not divide, an array from
% the host, and an array that a send copies, as it is sent again, to a
% process of the same kernel. The kind runs a loop, so that it has worker
% teams. Each array's weighted sum is the same at the host, here, and on
% the GPU, in process-bulk-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand: each line is the weighted sum below, worked out for the array.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -O2 -fopenmp -fdo-loops -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_bulk.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, int16, int32, int8, list, process,
    string.

    % The sum of each element's integer times 1 + its index mod 7.
:- func sum8(array(int8), int, int) = int.
sum8(A, I, S) = ( if I < array.size(A) then
    sum8(A, I + 1, S + int8.to_int(array.lookup(A, I)) * (1 + I mod 7))
    else S ).

:- func sum16(array(int16), int, int) = int.
sum16(A, I, S) = ( if I < array.size(A) then
    sum16(A, I + 1, S + int16.to_int(array.lookup(A, I)) * (1 + I mod 7))
    else S ).

:- func sum32(array(int32), int, int) = int.
sum32(A, I, S) = ( if I < array.size(A) then
    sum32(A, I + 1, S + int32.to_int(array.lookup(A, I)) * (1 + I mod 7))
    else S ).

:- func sum64(array(int), int, int) = int.
sum64(A, I, S) = ( if I < array.size(A) then
    sum64(A, I + 1, S + array.lookup(A, I) * (1 + I mod 7))
    else S ).

:- type rec
    --->    rec(int, float, int).

    % The same for records: the first field times 1 + the index mod 7, the
    % second's double as an integer, and the third times 1 + the index mod 5.
:- func sumr(array(rec), int, int) = int.
sumr(A, I, S) = ( if I < array.size(A) then
    sumr(A, I + 1, S + weigh(array.lookup(A, I), I))
    else S ).

:- func weigh(rec, int) = int.
weigh(rec(X, Y, Z), I) =
    X * (1 + I mod 7) + float.truncate_to_int(2.0 * Y) + Z * (1 + I mod 5).

:- process bulk(mailbox(int), mailbox(array(int))) receives array(int)
    on gpu.

bulk(Report, Summer, Self, !IO) :-
    % A loop, so that the kind has worker teams.
    array.init(64, 0, Z0),
    %$omp distribute parallel do
    ( for(I, 0, 63), fromto(Z0, Z1, Z2, Z) do
        array.set(I, I, Z1, Z2)
    ),
    send(Report, sum64(Z, 0, 0), !IO),
    A8a = array.init(100003, -3i8),
    send(Report, sum8(A8a, 0, 0), !IO),
    array.init(100003, -3i8, F0),
    array.fill(5i8, F0, A8b),
    send(Report, sum8(A8b, 0, 0), !IO),
    array.init(100003, 5i8, R0),
    array.resize(200005, 9i8, R0, A8c),
    send(Report, sum8(A8c, 0, 0), !IO),
    array.init(50001, -300i16, A16),
    array.resize(70003, 17i16, A16, A16b),
    send(Report, sum16(A16b, 0, 0), !IO),
    A32 = array.init(30001, 70000i32),
    send(Report, sum32(A32, 0, 0), !IO),
    A64 = array.init(20001, 123456789),
    B64 = array.copy(A64),
    array.set(77, 5, B64, C64),
    send(Report, sum64(A64, 0, 0), !IO),
    send(Report, sum64(C64, 0, 0), !IO),
    array.init(30001, rec(3, 0.5, -2), Ra),
    send(Report, sumr(Ra, 0, 0), !IO),
    array.resize(40003, rec(-1, 2.0, 7), array.copy(Ra), Rb),
    send(Report, sumr(Rb, 0, 0), !IO),
    receive(Self, any, after(10000), R, !IO),
    ( if R = message(H) then
        send(Report, sum64(H, 0, 0), !IO)
    else
        send(Report, -1, !IO)
    ),
    send(Summer, A64, !IO),
    send(Summer, A64, !IO).

    % Takes two arrays and reports their weighted sums.
:- process summer(mailbox(int)) receives array(int) on gpu.

summer(Report, Self, !IO) :-
    take(2, Report, Self, !IO).

:- pred take(int::in, mailbox(int)::in, mailbox(array(int))::in,
    io::di, io::uo) is det.

take(N, Report, Self, !IO) :-
    ( if N > 0 then
        receive(Self, any, after(10000), R, !IO),
        ( if R = message(A) then
            send(Report, sum64(A, 0, 0), !IO)
        else
            send(Report, -1, !IO)
        ),
        take(N - 1, Report, Self, !IO)
    else
        true
    ).

:- pred print_reports(int::in, mailbox(int)::in, io::di, io::uo) is det.

print_reports(N, Reports, !IO) :-
    ( if N > 0 then
        receive(Reports, any, after(20000), R, !IO),
        ( if R = message(V) then
            io.format("%d\n", [i(V)], !IO)
        else
            io.write_string("timed out\n", !IO)
        ),
        print_reports(N - 1, Reports, !IO)
    else
        true
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Reports, !IO),
    new_mailbox(Sums, !IO),
    spawn(Place, summer(Sums), S, !IO),
    spawn(Place, bulk(Reports, S), B, !IO),
    send(B, array.init(20001, -7), !IO),
    print_reports(11, Reports, !IO),
    print_reports(2, Sums, !IO).
% CHECK:8127
% CHECK-NEXT:-1200027
% CHECK-NEXT:2000045
% CHECK-NEXT:5600117
% CHECK-NEXT:-58641166
% CHECK-NEXT:8400070000
% CHECK-NEXT:9876419663211
% CHECK-NEXT:9876296206427
% CHECK-NEXT:210002
% CHECK-NEXT:420039
% CHECK-NEXT:-559993
% CHECK-NEXT:9876419663211
% CHECK-NEXT:9876419663211
% FREED-NEXT:Merkurius runtime: 0 messages are neither read nor freed
% FREED-NEXT:Merkurius runtime: 0 chunks of the devices' arenas and 0 rings are allocated
% FREED-NEXT:Merkurius runtime: 0 blocks of messages between GPU processes are allocated
