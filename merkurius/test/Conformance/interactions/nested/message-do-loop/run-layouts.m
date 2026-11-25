% MANUAL: Discriminated unions
% DESIGN: Processes and mailboxes
% DESIGN: Loops
% Interaction: a type of more functors than a pointer has tags, in a do loop
% that makes, gathers and carries the values, in messages sent to another
% process and back.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/nested_message_do_loop_layouts_translation.m
% MODULES: Inputs/nested_message_do_loop_layouts_lib.m
% FILES: Inputs/nested_message_do_loop_layouts_lib_translation.m
%
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_message_do_loop_layouts_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_message_do_loop_layouts_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0)) | last seven(three(10.5, one(21), 63), four(x, 441)) | seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | last seven(three(11.0, one(22), 66), eight(c0))
% OUTPUT-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | last seven(three(11.0, one(22), 66), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | last seven(three(11.5, one(23), 69), two(23, s23))
% OUTPUT-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | last seven(three(11.5, one(23), 69), two(23, s23)) | seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | last seven(three(12.0, one(24), 72), eight(c0))
% OUTPUT-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | last seven(three(12.0, one(24), 72), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) / seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | last seven(three(12.5, one(25), 75), two(25, s25))
% OUTPUT-NOT:{{.}}
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
