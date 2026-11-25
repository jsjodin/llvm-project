% MANUAL: Calling higher-order terms
% DESIGN: Processes and mailboxes
% MANUAL: Creating higher-order terms
% Interaction: closures stored in a term and called, in closures that make,
% capture and use the values, in messages sent to another process and back.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% TRANSLATION: Inputs/nested_message_closure_closures_as_data_translation.m
% MODULES: Inputs/nested_message_closure_closures_as_data_lib.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/nested_message_closure_closures_as_data_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/nested_message_closure_closures_as_data_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 11 3/1 / 11 3/1 | 11 3/1 | 21 5/2 | 21 5/2 / 21 5/2 | 21 5/2 | 31 7/3
% OUTPUT-NEXT:2: 21 5/2 / 21 5/2 | 21 5/2 | 31 7/3 | 31 7/3 / 31 7/3 | 31 7/3 | 41 9/4
% OUTPUT-NEXT:3: 31 7/3 / 31 7/3 | 31 7/3 | 41 9/4 | 41 9/4 / 41 9/4 | 41 9/4 | 51 11/5
% OUTPUT-NEXT:4: 41 9/4 / 41 9/4 | 41 9/4 | 51 11/5 | 51 11/5 / 51 11/5 | 51 11/5 | 61 13/6
% OUTPUT-NOT:{{.}}
:- module nested_message_closure_closures_as_data.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_message_closure_closures_as_data_lib, process, string.

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
