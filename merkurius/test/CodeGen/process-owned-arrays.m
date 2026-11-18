% Arrays that a GPU process's procedures take by mode di are freed by the
% callee (Frees.cpp, owned inputs): a caller does not use them again. A
% pass takes its vault by di, reads it, makes the next one and calls itself
% with it last, as a tracking pass with its extras does; with no owned
% inputs, every pass's array stayed until the process ended, as a tail
% call's arguments are not freed by its caller. A resize that clobbers its
% array frees the old one after it, and a procedure that resizes an array
% it takes by di and gives back by uo leaves the new one to its caller,
% which frees it. The kind runs at the host here, and on the GPU in
% process-owned-arrays-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand: passes(P, N) gives the sum over passes of N * pass, and grow the
% sum of the indices that it sets.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_owned_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process, string.

:- type job
    --->    passes(int, int)
    ;       grow(int)
    ;       stop.

:- process worker(mailbox(int)) receives job on gpu.

worker(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(passes(P, N)) then
        array.init(N, P, Vault),
        passes(P, N, Vault, 0, Sum),
        send(Reply, Sum, !IO),
        worker(Reply, Self, !IO)
    else if R = message(grow(N)) then
        array.init(1, 0, A0),
        grow(0, N, A0, A),
        send(Reply, sum(A, 0, N, 0), !IO),
        worker(Reply, Self, !IO)
    else
        true
    ).

    % passes(P, N, Vault, !Sum): the sum of the vaults of passes P down to 1,
    % each N elements of its pass's number.
:- pred passes(int::in, int::in, array(int)::array_di, int::in, int::out)
    is det.

passes(P, N, Vault, !Sum) :-
    ( if P > 0 then
        !:Sum = !.Sum + sum(Vault, 0, N, 0),
        array.init(N, P - 1, Next),
        passes(P - 1, N, Next, !Sum)
    else
        true
    ).

    % grow(I, N, !A): sets each index from I to N - 1 to itself, doubling A
    % when it is full.
:- pred grow(int::in, int::in, array(int)::array_di, array(int)::array_uo)
    is det.

grow(I, N, !A) :-
    ( if I < N then
        room(I, !A),
        !A ^ elem(I) := I,
        grow(I + 1, N, !A)
    else
        true
    ).

:- pred room(int::in, array(int)::array_di, array(int)::array_uo) is det.

room(I, !A) :-
    ( if I < array.size(!.A) then
        true
    else
        array.resize(2 * array.size(!.A), 0, !A)
    ).

:- func sum(array(int), int, int, int) = int.

sum(A, I, N, S) =
    ( if I < N then sum(A, I + 1, N, S + A ^ elem(I)) else S ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["host"] then host else gpu(0) ),
    new_mailbox(Me, !IO),
    new_mailbox(Downs, !IO),
    spawn(Place, worker(Me), W, !IO),
    monitor(W, Downs, id, _, !IO),
    list.foldl(ask(W, Me), [passes(16, 200000), grow(400000)], !IO),
    send(W, stop, !IO),
    receive(Downs, any, after(60000), _, !IO).

:- func id(down_info) = down_info.
id(D) = D.

:- pred ask(mailbox(job)::in, mailbox(int)::in, job::in, io::di, io::uo)
    is det.
ask(W, Me, Job, !IO) :-
    send(W, Job, !IO),
    receive(Me, any, after(60000), R, !IO),
    ( if R = message(V) then
        io.write_int(V, !IO),
        io.nl(!IO)
    else
        io.write_string("timeout\n", !IO)
    ).

% CHECK:27200000
% CHECK-NEXT:79999800000
