% MANUAL: Unique modes
% DESIGN: Processes and mailboxes
% MANUAL: Creating higher-order terms
% Interaction: an array updated in place in unique modes, in closures that
% make, capture and use the values, in another process, which makes and uses
% the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% TRANSLATION: Inputs/nested_host_process_closure_unique_modes_translation.m
% MODULES: Inputs/nested_host_process_closure_unique_modes_lib.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/nested_host_process_closure_unique_modes_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/nested_host_process_closure_unique_modes_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 0 1 4 9 / 0 1 4 9 | 0 1 4 9 | 0 1 4 9 16 | 0 1 4 9 16 / 0 1 4 9 16 | 0 1 4 9 16 | 0 1 4 9 16 25
% OUTPUT-NEXT:2: 0 1 4 9 16 / 0 1 4 9 16 | 0 1 4 9 16 | 0 1 4 9 16 25 | 0 1 4 9 16 25 / 0 1 4 9 16 25 | 0 1 4 9 16 25 | 0 1 4 9 16 25 36
% OUTPUT-NEXT:3: 0 1 4 9 16 25 / 0 1 4 9 16 25 | 0 1 4 9 16 25 | 0 1 4 9 16 25 36 | 0 1 4 9 16 25 36 / 0 1 4 9 16 25 36 | 0 1 4 9 16 25 36 | 0 1 4 9 16 25 36 49
% OUTPUT-NEXT:4: 0 1 4 9 16 25 36 / 0 1 4 9 16 25 36 | 0 1 4 9 16 25 36 | 0 1 4 9 16 25 36 49 | 0 1 4 9 16 25 36 49 / 0 1 4 9 16 25 36 49 | 0 1 4 9 16 25 36 49 | 0 1 4 9 16 25 36 49 64
% OUTPUT-NOT:{{.}}
:- module nested_host_process_closure_unique_modes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_host_process_closure_unique_modes_lib, process, string.

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
