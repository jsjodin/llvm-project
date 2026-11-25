% MANUAL: Equivalence types
% DESIGN: Processes and mailboxes
% Interaction: a type defined as another, generic one, in messages sent to
% another process and back.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% TRANSLATION: Inputs/message_equivalence_types_translation.m
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 1: n1 | 2: n1 n2
% OUTPUT-NEXT:2: 2: n1 n2 | 3: n1 n2 n3
% OUTPUT-NEXT:3: 3: n1 n2 n3 | 4: n1 n2 n3 n4
% OUTPUT-NEXT:4: 4: n1 n2 n3 n4 | 5: n1 n2 n3 n4 n5
% OUTPUT-NOT:{{.}}
:- module message_equivalence_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, pair, process, string.

:- type name == string.
:- type fdata == pair(int, list(name)).

:- func make(int) = fdata.
:- func use(fdata) = string.

make(N) = N - list.map((func(I) = "n" ++ string.int_to_string(I)),
    1 .. N).

use(K - Names) = string.int_to_string(K) ++ ": " ++
    string.join_list(" ", Names).

    % The values as messages: copied into another process's heap, and back.
:- pred echo(int::in, mailbox(fdata)::in, mailbox(fdata)::in,
    io::di, io::uo) is det.

echo(Left, Back, Self, !IO) :-
    ( if Left = 0 then
        true
    else
        receive(Self, any, infinity, R, !IO),
        (
            R = message(V),
            send(Back, V, !IO),
            echo(Left - 1, Back, Self, !IO)
        ;
            R = timeout
        )
    ).

:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, S, !IO) :-
    new_mailbox(Back, !IO),
    spawn(echo(2, Back), Echo, !IO),
    send(Echo, make(N), !IO),
    send(Echo, make(N + 1), !IO),
    back(Back, S1, !IO),
    back(Back, S2, !IO),
    S = S1 ++ " | " ++ S2.

:- pred back(mailbox(fdata)::in, string::out, io::di, io::uo) is det.

back(Back, S, !IO) :-
    receive(Back, any, infinity, R, !IO),
    ( R = message(V), S = use(V) ; R = timeout, S = "timeout" ).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    run(N, S, !IO),
    io.format("%d: %s\n", [i(N), s(S)], !IO).
