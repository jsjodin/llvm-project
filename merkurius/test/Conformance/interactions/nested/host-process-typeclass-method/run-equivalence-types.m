% MANUAL: Equivalence types
% DESIGN: Processes and mailboxes
% MANUAL: Type classes
% Interaction: a type defined as another, generic one, in instance methods
% that make and use the values, in another process, which makes and uses the
% values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% TRANSLATION: Inputs/nested_host_process_typeclass_method_equivalence_types_translation.m
% MODULES: Inputs/nested_host_process_typeclass_method_equivalence_types_lib.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/nested_host_process_typeclass_method_equivalence_types_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/nested_host_process_typeclass_method_equivalence_types_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 1: n1 / 1: n1 | offset 2: n1 n2 | 2: n1 n2 / 2: n1 n2 | offset 3: n1 n2 n3
% OUTPUT-NEXT:2: 2: n1 n2 / 2: n1 n2 | offset 3: n1 n2 n3 | 3: n1 n2 n3 / 3: n1 n2 n3 | offset 4: n1 n2 n3 n4
% OUTPUT-NEXT:3: 3: n1 n2 n3 / 3: n1 n2 n3 | offset 4: n1 n2 n3 n4 | 4: n1 n2 n3 n4 / 4: n1 n2 n3 n4 | offset 5: n1 n2 n3 n4 n5
% OUTPUT-NEXT:4: 4: n1 n2 n3 n4 / 4: n1 n2 n3 n4 | offset 5: n1 n2 n3 n4 n5 | 5: n1 n2 n3 n4 n5 / 5: n1 n2 n3 n4 n5 | offset 6: n1 n2 n3 n4 n5 n6
% OUTPUT-NOT:{{.}}
:- module nested_host_process_typeclass_method_equivalence_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_host_process_typeclass_method_equivalence_types_lib, process, string.

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
