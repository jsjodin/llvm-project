% MANUAL: Discriminated unions
% DESIGN: Processes and mailboxes
% DESIGN: Loops
% DESIGN: Sized floating point types
% Interaction: a type of more functors than a pointer has tags, in a do loop
% that makes, gathers and carries the values, in messages sent to another
% process and back.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/nested_message_do_loop_layouts_translation.mt
% MODULES: Inputs/nested_message_do_loop_layouts_lib.mt
% FILES: Inputs/nested_message_do_loop_layouts_lib_translation.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: %instantiate --keep-name %S/Inputs/nested_message_do_loop_layouts_lib.mt float %t.float.d
% RUN: merkc -O0 -fdo-loops -I %t.float.d %t.float.d/nested_message_do_loop_layouts_float.m %t.float.d/nested_message_do_loop_layouts_lib.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %t.float.d %t.float.d/nested_message_do_loop_layouts_float.m %t.float.d/nested_message_do_loop_layouts_lib.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0)) | last seven(three(10.5, one(21), 63), four(x, 441)) | seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | last seven(three(11.0, one(22), 66), eight(c0))
% OUTPUT-FLOAT-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | last seven(three(11.0, one(22), 66), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | last seven(three(11.5, one(23), 69), two(23, s23))
% OUTPUT-FLOAT-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | last seven(three(11.5, one(23), 69), two(23, s23)) | seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | last seven(three(12.0, one(24), 72), eight(c0))
% OUTPUT-FLOAT-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | last seven(three(12.0, one(24), 72), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) / seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | last seven(three(12.5, one(25), 75), two(25, s25))
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: %instantiate --keep-name %S/Inputs/nested_message_do_loop_layouts_lib.mt float32 %t.float32.d
% RUN: merkc -O0 -fdo-loops -I %t.float32.d %t.float32.d/nested_message_do_loop_layouts_float32.m %t.float32.d/nested_message_do_loop_layouts_lib.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %t.float32.d %t.float32.d/nested_message_do_loop_layouts_float32.m %t.float32.d/nested_message_do_loop_layouts_lib.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0)) | last seven(three(10.5, one(21), 63), four(x, 441)) | seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | last seven(three(11.0, one(22), 66), eight(c0))
% OUTPUT-FLOAT32-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | last seven(three(11.0, one(22), 66), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | last seven(three(11.5, one(23), 69), two(23, s23))
% OUTPUT-FLOAT32-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | last seven(three(11.5, one(23), 69), two(23, s23)) | seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | last seven(three(12.0, one(24), 72), eight(c0))
% OUTPUT-FLOAT32-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | last seven(three(12.0, one(24), 72), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) / seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | last seven(three(12.5, one(25), 75), two(25, s25))
% OUTPUT-FLOAT32-NOT:{{.}}
:- module nested_message_do_loop_layouts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_message_do_loop_layouts_lib, process, string.

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
