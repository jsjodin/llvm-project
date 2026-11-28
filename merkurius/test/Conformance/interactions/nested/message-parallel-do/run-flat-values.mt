% MANUAL: Discriminated unions
% DESIGN: Processes and mailboxes
% DESIGN: Loops / Under a parallel directive
% DESIGN: Sized floating point types
% Interaction: an enumeration, a notag and a record of numbers, in a
% parallel loop's iterations, which make and use the values, in messages
% sent to another process and back.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/nested_message_parallel_do_flat_values_translation.mt
% MODULES: Inputs/nested_message_parallel_do_flat_values_lib.mt
% FILES: Inputs/nested_message_parallel_do_flat_values_lib_translation.mt
%
% REQUIRES: libomp
% RUN: %instantiate %s float %t.float.d
% RUN: %instantiate --keep-name %S/Inputs/nested_message_parallel_do_flat_values_lib.mt float %t.float.d
% RUN: merkc -O0 -fdo-loops -fopenmp -I %t.float.d %t.float.d/nested_message_parallel_do_flat_values_float.m %t.float.d/nested_message_parallel_do_flat_values_lib.m %libomp-flags -o %t.float.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %t.float.d %t.float.d/nested_message_parallel_do_flat_values_float.m %t.float.d/nested_message_parallel_do_flat_values_lib.m %libomp-flags -o %t.float.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %t.float.d %t.float.d/nested_message_parallel_do_flat_values_float.m %t.float.d/nested_message_parallel_do_flat_values_lib.m -o %t.float.0s
% RUN: %t.float.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %t.float.d %t.float.d/nested_message_parallel_do_flat_values_float.m %t.float.d/nested_message_parallel_do_flat_values_lib.m -o %t.float.2s
% RUN: %t.float.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: green 100 -1 1000 0.25 / green 100 -1 1000 0.25 | blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 200 -2 2000 0.5 / blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25
% OUTPUT-FLOAT-NEXT:2: blue 200 -2 2000 0.5 / blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5
% OUTPUT-FLOAT-NEXT:3: red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 | green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 | green 700 -7 7000 1.75
% OUTPUT-FLOAT-NEXT:4: green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 | green 700 -7 7000 1.75 | blue 500 -5 5000 1.25 / blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 | green 700 -7 7000 1.75 | blue 800 -8 8000 2.0
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: %instantiate --keep-name %S/Inputs/nested_message_parallel_do_flat_values_lib.mt float32 %t.float32.d
% RUN: merkc -O0 -fdo-loops -fopenmp -I %t.float32.d %t.float32.d/nested_message_parallel_do_flat_values_float32.m %t.float32.d/nested_message_parallel_do_flat_values_lib.m %libomp-flags -o %t.float32.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %t.float32.d %t.float32.d/nested_message_parallel_do_flat_values_float32.m %t.float32.d/nested_message_parallel_do_flat_values_lib.m %libomp-flags -o %t.float32.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %t.float32.d %t.float32.d/nested_message_parallel_do_flat_values_float32.m %t.float32.d/nested_message_parallel_do_flat_values_lib.m -o %t.float32.0s
% RUN: %t.float32.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %t.float32.d %t.float32.d/nested_message_parallel_do_flat_values_float32.m %t.float32.d/nested_message_parallel_do_flat_values_lib.m -o %t.float32.2s
% RUN: %t.float32.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: green 100 -1 1000 0.25 / green 100 -1 1000 0.25 | blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 200 -2 2000 0.5 / blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25
% OUTPUT-FLOAT32-NEXT:2: blue 200 -2 2000 0.5 / blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5
% OUTPUT-FLOAT32-NEXT:3: red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 | green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 | green 700 -7 7000 1.75
% OUTPUT-FLOAT32-NEXT:4: green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 | green 700 -7 7000 1.75 | blue 500 -5 5000 1.25 / blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 | green 700 -7 7000 1.75 | blue 800 -8 8000 2.0
% OUTPUT-FLOAT32-NOT:{{.}}
:- module nested_message_parallel_do_flat_values.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_message_parallel_do_flat_values_lib, process, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

    % The values as messages: copied into another process's heap, and back.
:- pred echo(int::in, mailbox(nested)::in, mailbox(nested)::in,
    io::di, io::uo) is det.

echo(Left, Back, Self, !IO) :-
    ( if Left = 0 then
        true
    else
        process.receive(Self, any, infinity, R, !IO),
        (
            R = message(V),
            process.send(Back, V, !IO),
            echo(Left - 1, Back, Self, !IO)
        ;
            R = timeout
        )
    ).

:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, S, !IO) :-
    new_mailbox(Back, !IO),
    spawn(echo(2, Back), Echo, !IO),
    process.send(Echo, make_nested(N), !IO),
    process.send(Echo, make_nested(N + 1), !IO),
    back(Back, S1, !IO),
    back(Back, S2, !IO),
    S = S1 ++ " | " ++ S2.

:- pred back(mailbox(nested)::in, string::out, io::di, io::uo) is det.

back(Back, S, !IO) :-
    process.receive(Back, any, infinity, R, !IO),
    ( R = message(V), S = use_nested(V) ; R = timeout, S = "timeout" ).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    run(N, S, !IO),
    io.format("%d: %s\n", [i(N), s(S)], !IO).
