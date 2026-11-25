% MANUAL: Field access functions
% DESIGN: GPU processes
% Interaction: field access and update functions, in a GPU process, which
% makes and uses the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fprocess-declarations
% TRANSLATION: Inputs/gpu_process_field_access_translation.m
%
% RUN: merkc -O0 -fprocess-declarations %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 10199 | 20298
% OUTPUT-NEXT:2: 20298 | 30397
% OUTPUT-NEXT:3: 30397 | 40496
% OUTPUT-NEXT:4: 40496 | 50595
% OUTPUT-NOT:{{.}}
:- module gpu_process_field_access.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

:- type fdata
    --->    account(
                id :: int,
                balance :: int,
                limit :: int
            ).

:- func make(int) = fdata.
:- func use(fdata) = int.

make(N) = account(N, N * 10, 50).

use(A0) = A ^ id * 10000 + A ^ balance * 10 + A ^ limit :-
    A1 = A0 ^ balance := A0 ^ balance + 5,
    A = A1 ^ limit := A1 ^ limit - A0 ^ id.

    % A process kind on gpu, which makes and uses the values where it runs:
    % at the host, or on the GPU when the program's argument is gpu
    % (run-*-gpu.test).
:- type request
    --->    request(int, mailbox(int))
    ;       stop.

:- process worker receives request on gpu.

worker(Self, !IO) :-
    process.receive(Self, any, infinity, R, !IO),
    (
        R = message(request(K, Reply)),
        process.send(Reply, use(make(K)), !IO),
        worker(Self, !IO)
    ;
        R = message(stop)
    ;
        R = timeout
    ).

:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, S, !IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Replies, !IO),
    spawn(Place, worker, Worker, !IO),
    process.send(Worker, request(N, Replies), !IO),
    process.send(Worker, request(N + 1, Replies), !IO),
    process.send(Worker, stop, !IO),
    reply(Replies, S1, !IO),
    reply(Replies, S2, !IO),
    S = S1 ++ " | " ++ S2.

:- pred reply(mailbox(int)::in, string::out, io::di, io::uo) is det.

reply(Replies, S, !IO) :-
    process.receive(Replies, any, infinity, R, !IO),
    ( R = message(K), S = string.int_to_string(K)
    ; R = timeout, S = "timeout"
    ).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    run(N, S, !IO),
    io.format("%d: %s\n", [i(N), s(S)], !IO).
