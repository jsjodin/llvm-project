% MANUAL: Discriminated unions
% DESIGN: Processes and mailboxes
% DESIGN: Loops
% Interaction: an enumeration, a notag and a record of numbers, in a do loop
% that makes, gathers and carries the values, in another process, which
% makes and uses the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/nested_host_process_do_loop_flat_values_translation.m
% MODULES: Inputs/nested_host_process_do_loop_flat_values_lib.m
% FILES: Inputs/nested_host_process_do_loop_flat_values_lib_translation.m
%
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_host_process_do_loop_flat_values_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_host_process_do_loop_flat_values_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: green 100 -1 1000 0.25 / green 100 -1 1000 0.25 | blue 200 -2 2000 0.5 | last red 2100 -21 21000 5.25 | blue 200 -2 2000 0.5 / blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | last green 2200 -22 22000 5.5
% OUTPUT-NEXT:2: blue 200 -2 2000 0.5 / blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | last green 2200 -22 22000 5.5 | red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | last blue 2300 -23 23000 5.75
% OUTPUT-NEXT:3: red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | last blue 2300 -23 23000 5.75 | green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | last red 2400 -24 24000 6.0
% OUTPUT-NEXT:4: green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | last red 2400 -24 24000 6.0 | blue 500 -5 5000 1.25 / blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 | last green 2500 -25 25000 6.25
% OUTPUT-NOT:{{.}}
:- module nested_host_process_do_loop_flat_values.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_host_process_do_loop_flat_values_lib, process, string.

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
