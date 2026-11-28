% MANUAL: Discriminated unions
% DESIGN: Processes and mailboxes
% DESIGN: Loops / Under a parallel directive
% DESIGN: Sized floating point types
% Interaction: a type of more functors than a pointer has tags, in a
% parallel loop's iterations, which make and use the values, in another
% process, which makes and uses the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/nested_host_process_parallel_do_layouts_translation.mt
% MODULES: Inputs/nested_host_process_parallel_do_layouts_lib.mt
% FILES: Inputs/nested_host_process_parallel_do_layouts_lib_translation.mt
%
% REQUIRES: libomp
% RUN: %instantiate %s float %t.float.d
% RUN: %instantiate --keep-name %S/Inputs/nested_host_process_parallel_do_layouts_lib.mt float %t.float.d
% RUN: merkc -O0 -fdo-loops -fopenmp -I %t.float.d %t.float.d/nested_host_process_parallel_do_layouts_float.m %t.float.d/nested_host_process_parallel_do_layouts_lib.m %libomp-flags -o %t.float.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %t.float.d %t.float.d/nested_host_process_parallel_do_layouts_float.m %t.float.d/nested_host_process_parallel_do_layouts_lib.m %libomp-flags -o %t.float.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %t.float.d %t.float.d/nested_host_process_parallel_do_layouts_float.m %t.float.d/nested_host_process_parallel_do_layouts_lib.m -o %t.float.0s
% RUN: %t.float.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %t.float.d %t.float.d/nested_host_process_parallel_do_layouts_float.m %t.float.d/nested_host_process_parallel_do_layouts_lib.m -o %t.float.2s
% RUN: %t.float.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0))
% OUTPUT-FLOAT-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | seven(three(3.5, one(7), 21), two(7, s7))
% OUTPUT-FLOAT-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | seven(three(3.5, one(7), 21), two(7, s7)) | seven(three(2.5, one(5), 15), two(5, s5)) / seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | seven(three(3.5, one(7), 21), two(7, s7)) | seven(three(4.0, one(8), 24), eight(c0))
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: %instantiate --keep-name %S/Inputs/nested_host_process_parallel_do_layouts_lib.mt float32 %t.float32.d
% RUN: merkc -O0 -fdo-loops -fopenmp -I %t.float32.d %t.float32.d/nested_host_process_parallel_do_layouts_float32.m %t.float32.d/nested_host_process_parallel_do_layouts_lib.m %libomp-flags -o %t.float32.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %t.float32.d %t.float32.d/nested_host_process_parallel_do_layouts_float32.m %t.float32.d/nested_host_process_parallel_do_layouts_lib.m %libomp-flags -o %t.float32.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %t.float32.d %t.float32.d/nested_host_process_parallel_do_layouts_float32.m %t.float32.d/nested_host_process_parallel_do_layouts_lib.m -o %t.float32.0s
% RUN: %t.float32.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %t.float32.d %t.float32.d/nested_host_process_parallel_do_layouts_float32.m %t.float32.d/nested_host_process_parallel_do_layouts_lib.m -o %t.float32.2s
% RUN: %t.float32.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT32-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0))
% OUTPUT-FLOAT32-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | seven(three(3.5, one(7), 21), two(7, s7))
% OUTPUT-FLOAT32-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | seven(three(3.5, one(7), 21), two(7, s7)) | seven(three(2.5, one(5), 15), two(5, s5)) / seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | seven(three(3.5, one(7), 21), two(7, s7)) | seven(three(4.0, one(8), 24), eight(c0))
% OUTPUT-FLOAT32-NOT:{{.}}
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
