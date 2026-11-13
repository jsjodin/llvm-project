% The blocks of a GPU process (runtime/device/device.cpp, Block): its cells
% and arrays come from libc's allocator, a large array's from the image's
% cache of large blocks, which a later process reuses, and all go when the
% process ends. A maker takes a size, makes an array of that many ints, each
% its index, and answers with their sum; makers run one after another, with
% arrays from ten elements to 3.2 MB, the large ones twice, and one maker
% takes several sizes before it ends. The kind runs at the host here, and on
% the GPU in process-blocks-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand: each sum is N * (N - 1) / 2.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_blocks.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process, string.

    % Answers each size with the sum of an array of that many indices, until
    % a size of 0.
:- process maker(mailbox(int)) receives int on gpu.

maker(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N), N > 0 then
        array.init(N, 0, A0),
        fill(0, N, A0, A),
        send(Reply, sum(A, 0, N, 0), !IO),
        maker(Reply, Self, !IO)
    else
        true
    ).

:- pred fill(int::in, int::in, array(int)::array_di, array(int)::array_uo)
    is det.
fill(I, N, !A) :-
    ( if I < N then
        !A ^ elem(I) := I,
        fill(I + 1, N, !A)
    else
        true
    ).

:- func sum(array(int), int, int, int) = int.
sum(A, I, N, S) = ( if I < N then sum(A, I + 1, N, S + A ^ elem(I)) else S ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["host"] then host else gpu(0) ),
    new_mailbox(Me, !IO),
    new_mailbox(Downs, !IO),
    list.foldl(one_maker(Place, Me, Downs),
        [[10], [1000], [200000], [200000], [400000], [10, 400000, 300000]],
        !IO).

    % A maker for the sizes, which ends after them; the next starts once it
    % has ended.
:- pred one_maker(place::in, mailbox(int)::in, mailbox(down_info)::in,
    list(int)::in, io::di, io::uo) is det.
one_maker(Place, Me, Downs, Sizes, !IO) :-
    spawn(Place, maker(Me), M, !IO),
    monitor(M, Downs, id, _, !IO),
    list.foldl(ask(M, Me), Sizes, !IO),
    send(M, 0, !IO),
    receive(Downs, any, after(60000), _, !IO).

:- func id(down_info) = down_info.
id(D) = D.

:- pred ask(mailbox(int)::in, mailbox(int)::in, int::in, io::di, io::uo)
    is det.
ask(M, Me, N, !IO) :-
    send(M, N, !IO),
    receive(Me, any, after(60000), R, !IO),
    ( if R = message(Sum), Sum = N * (N - 1) / 2 then
        io.format("%d: right\n", [i(N)], !IO)
    else
        io.format("%d: wrong\n", [i(N)], !IO)
    ).

% CHECK:10: right
% CHECK-NEXT:1000: right
% CHECK-NEXT:200000: right
% CHECK-NEXT:200000: right
% CHECK-NEXT:400000: right
% CHECK-NEXT:10: right
% CHECK-NEXT:400000: right
% CHECK-NEXT:300000: right
