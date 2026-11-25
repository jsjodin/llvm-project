% MANUAL: Determinism categories
% DESIGN: GPU processes
% Interaction: nondet and semidet code, in a GPU process, which makes and
% uses the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fprocess-declarations
% TRANSLATION: Inputs/gpu_process_determinism_translation.m
%
% RUN: merkc -O0 -fprocess-declarations %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 301 | 302
% OUTPUT-NEXT:2: 302 | 302
% OUTPUT-NEXT:3: 302 | 303
% OUTPUT-NEXT:4: 303 | 303
% OUTPUT-NOT:{{.}}
:- module gpu_process_determinism.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

:- type fdata
    --->    range(int, int).

:- func make(int) = fdata.
:- func use(fdata) = int.

make(N) = range(1, N + 2).

    % The first number over 2 in the range, by a semidet search, and how
    % many are even, by a nondet one, aggregated.
use(range(Lo, Hi)) = First * 100 + Evens :-
    ( if first_over(Lo, Hi, 2, F) then First = F else First = 0 ),
    Evens = count_evens(Lo, Hi, 0).

:- pred first_over(int::in, int::in, int::in, int::out) is semidet.

first_over(I, Hi, Limit, F) :-
    I =< Hi,
    ( if I > Limit then
        F = I
    else
        first_over(I + 1, Hi, Limit, F)
    ).

:- pred even_in(int::in, int::in, int::out) is nondet.

even_in(Lo, Hi, X) :-
    in_range(Lo, Hi, X),
    X mod 2 = 0.

:- pred in_range(int::in, int::in, int::out) is nondet.

in_range(Lo, Hi, X) :-
    Lo =< Hi,
    ( X = Lo
    ; in_range(Lo + 1, Hi, X)
    ).

:- func count_evens(int, int, int) = int.

count_evens(Lo, Hi, N0) = N :-
    promise_equivalent_solutions [N] (
        ( if even_in(Lo, Hi, X), X > N0 then
            N = 1 + count_evens(X + 1, Hi, N0)
        else
            N = 0
        )
    ).

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
