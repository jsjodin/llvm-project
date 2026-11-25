% MANUAL: Discriminated unions
% DESIGN: Processes and mailboxes
% DESIGN: Loops / Under a parallel directive
% Interaction: a type of more functors than a pointer has tags, in a
% parallel loop's iterations, which make and use the values, in another
% process, which makes and uses the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/nested_host_process_parallel_do_layouts_translation.m
% MODULES: Inputs/nested_host_process_parallel_do_layouts_lib.m
% FILES: Inputs/nested_host_process_parallel_do_layouts_lib_translation.m
%
% REQUIRES: libomp
% RUN: merkc -O0 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_host_process_parallel_do_layouts_lib.m %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_host_process_parallel_do_layouts_lib.m %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_host_process_parallel_do_layouts_lib.m -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_host_process_parallel_do_layouts_lib.m -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0))
% OUTPUT-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | seven(three(3.5, one(7), 21), two(7, s7))
% OUTPUT-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | seven(three(3.5, one(7), 21), two(7, s7)) | seven(three(2.5, one(5), 15), two(5, s5)) / seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | seven(three(3.5, one(7), 21), two(7, s7)) | seven(three(4.0, one(8), 24), eight(c0))
% OUTPUT-NOT:{{.}}
:- module nested_host_process_parallel_do_layouts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_host_process_parallel_do_layouts_lib, process, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

    % Another process makes and uses the values, in its own heap, and
    % sends the text back.
:- pred worker(mailbox(string)::in, mailbox(int)::in, io::di, io::uo)
    is det.

worker(Replies, Self, !IO) :-
    process.receive(Self, any, infinity, R, !IO),
    (
        R = message(K),
        ( if K < 0 then
            true
        else
            process.send(Replies, use_nested(make_nested(K)), !IO),
            worker(Replies, Self, !IO)
        )
    ;
        R = timeout
    ).

:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, S, !IO) :-
    new_mailbox(Replies, !IO),
    spawn(worker(Replies), Worker, !IO),
    process.send(Worker, N, !IO),
    process.send(Worker, N + 1, !IO),
    process.send(Worker, -1, !IO),
    reply(Replies, S1, !IO),
    reply(Replies, S2, !IO),
    S = S1 ++ " | " ++ S2.

:- pred reply(mailbox(string)::in, string::out, io::di, io::uo) is det.

reply(Replies, S, !IO) :-
    process.receive(Replies, any, infinity, R, !IO),
    ( R = message(S) ; R = timeout, S = "timeout" ).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    run(N, S, !IO),
    io.format("%d: %s\n", [i(N), s(S)], !IO).
