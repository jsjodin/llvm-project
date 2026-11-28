% MANUAL: Discriminated unions
% DESIGN: GPU processes
% DESIGN: Sized floating point types
% Interaction: an enumeration, a notag and a record of numbers, in a GPU
% process, which makes and uses the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fprocess-declarations
% TRANSLATION: Inputs/gpu_process_flat_values_translation.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 -fprocess-declarations %t.float.d/gpu_process_flat_values_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %t.float.d/gpu_process_flat_values_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: 1126 | 2251
% OUTPUT-FLOAT-NEXT:2: 2251 | 3373
% OUTPUT-FLOAT-NEXT:3: 3373 | 4498
% OUTPUT-FLOAT-NEXT:4: 4498 | 5623
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 -fprocess-declarations %t.float32.d/gpu_process_flat_values_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %t.float32.d/gpu_process_flat_values_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: 1126 | 2251
% OUTPUT-FLOAT32-NEXT:2: 2251 | 3373
% OUTPUT-FLOAT32-NEXT:3: 3373 | 4498
% OUTPUT-FLOAT32-NEXT:4: 4498 | 5623
% OUTPUT-FLOAT32-NOT:{{.}}
:- module gpu_process_flat_values.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, int8, list, process, string, uint16.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type wrapped
    --->    wrapped(int).

    % A record of one functor of numbers, an enumeration and a notag.
:- type fdata
    --->    fdata(colour, wrapped, int8, uint16, @T@).

:- func make(int) = fdata.
:- func use(fdata) = int.

make(N) = fdata(
    ( if N mod 3 = 0 then red else if N mod 3 = 1 then green else blue ),
    wrapped(N * 100), int8.det_from_int(-N), uint16.det_from_int(N * 1000),
    @FROM_INT@(N) * 0.25@S@).

use(fdata(C, wrapped(W), I, U, F)) =
    colour_code(C) + W + int8.to_int(I) + uint16.to_int(U) +
    truncate_to_int(F * 100.0@S@).

:- func colour_code(colour) = int.

colour_code(red) = 1.
colour_code(green) = 2.
colour_code(blue) = 3.

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
