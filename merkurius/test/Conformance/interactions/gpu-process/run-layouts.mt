% MANUAL: Discriminated unions
% DESIGN: GPU processes
% DESIGN: Sized floating point types
% Interaction: a type of more functors than a pointer has tags, in a GPU
% process, which makes and uses the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fprocess-declarations
% TRANSLATION: Inputs/gpu_process_layouts_translation.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 -fprocess-declarations %t.float.d/gpu_process_layouts_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %t.float.d/gpu_process_layouts_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: 1 | 18
% OUTPUT-FLOAT-NEXT:2: 18 | 18
% OUTPUT-FLOAT-NEXT:3: 18 | 4123
% OUTPUT-FLOAT-NEXT:4: 4123 | 20
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 -fprocess-declarations %t.float32.d/gpu_process_layouts_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %t.float32.d/gpu_process_layouts_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: 1 | 18
% OUTPUT-FLOAT32-NEXT:2: 18 | 18
% OUTPUT-FLOAT32-NEXT:3: 18 | 4123
% OUTPUT-FLOAT32-NEXT:4: 4123 | 20
% OUTPUT-FLOAT32-NOT:{{.}}
:- module gpu_process_layouts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, list, process, string.

    % More functors with arguments than a pointer has tags, and constants,
    % of fixed size: a flat union.
:- type fdata
    --->    c0
    ;       c1
    ;       one(int)
    ;       two(int, int)
    ;       three(@T@, int)
    ;       four(int, int, int, int)
    ;       five(@T@)
    ;       six(int)
    ;       seven(int, @T@, int).

:- func make(int) = fdata.
:- func use(fdata) = int.

make(N) = D :-
    K = N mod 9,
    ( if K = 0 then D = c0
    else if K = 1 then D = one(N)
    else if K = 2 then D = two(N, -N)
    else if K = 3 then D = three(@FROM_INT@(N) / 2.0@S@, N)
    else if K = 4 then D = four(N, 1, 2, 3)
    else if K = 5 then D = five(@FROM_INT@(N) * 4.0@S@)
    else if K = 6 then D = six(N * N)
    else if K = 7 then D = seven(N, 0.5@S@, N)
    else D = c1
    ).

use(c0) = 1000.
use(c1) = 2000.
use(one(I)) = I.
use(two(I, J)) = I * 10 + J.
use(three(F, I)) = truncate_to_int(F * 10.0@S@) + I.
use(four(A, B, C, D)) = A * 1000 + B * 100 + C * 10 + D.
use(five(F)) = truncate_to_int(F).
use(six(I)) = I + 7.
use(seven(A, F, B)) = A + truncate_to_int(F * 100.0@S@) + B.

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
