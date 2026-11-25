% MANUAL: Equivalence types
% DESIGN: Processes and mailboxes
% MANUAL: Creating higher-order terms
% Interaction: a type defined as another, generic one, in closures that
% make, capture and use the values, in messages sent to another process and
% back.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% TRANSLATION: Inputs/nested_message_closure_equivalence_types_translation.m
% MODULES: Inputs/nested_message_closure_equivalence_types_lib.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/nested_message_closure_equivalence_types_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/nested_message_closure_equivalence_types_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 1: n1 / 1: n1 | 1: n1 | 2: n1 n2 | 2: n1 n2 / 2: n1 n2 | 2: n1 n2 | 3: n1 n2 n3
% OUTPUT-NEXT:2: 2: n1 n2 / 2: n1 n2 | 2: n1 n2 | 3: n1 n2 n3 | 3: n1 n2 n3 / 3: n1 n2 n3 | 3: n1 n2 n3 | 4: n1 n2 n3 n4
% OUTPUT-NEXT:3: 3: n1 n2 n3 / 3: n1 n2 n3 | 3: n1 n2 n3 | 4: n1 n2 n3 n4 | 4: n1 n2 n3 n4 / 4: n1 n2 n3 n4 | 4: n1 n2 n3 n4 | 5: n1 n2 n3 n4 n5
% OUTPUT-NEXT:4: 4: n1 n2 n3 n4 / 4: n1 n2 n3 n4 | 4: n1 n2 n3 n4 | 5: n1 n2 n3 n4 n5 | 5: n1 n2 n3 n4 n5 / 5: n1 n2 n3 n4 n5 | 5: n1 n2 n3 n4 n5 | 6: n1 n2 n3 n4 n5 n6
% OUTPUT-NOT:{{.}}
:- module nested_message_closure_equivalence_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_message_closure_equivalence_types_lib, process, string.

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
