% MANUAL: Discriminated unions
% DESIGN: Processes and mailboxes
% DESIGN: Sized floating point types
% Interaction: an enumeration, a notag and a record of numbers, in another
% process, which makes and uses the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% TRANSLATION: Inputs/host_process_flat_values_translation.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 %t.float.d/host_process_flat_values_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/host_process_flat_values_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: green 100 -1 1000 0.25 | blue 200 -2 2000 0.5
% OUTPUT-FLOAT-NEXT:2: blue 200 -2 2000 0.5 | red 300 -3 3000 0.75
% OUTPUT-FLOAT-NEXT:3: red 300 -3 3000 0.75 | green 400 -4 4000 1.0
% OUTPUT-FLOAT-NEXT:4: green 400 -4 4000 1.0 | blue 500 -5 5000 1.25
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 %t.float32.d/host_process_flat_values_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/host_process_flat_values_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: green 100 -1 1000 0.25 | blue 200 -2 2000 0.5
% OUTPUT-FLOAT32-NEXT:2: blue 200 -2 2000 0.5 | red 300 -3 3000 0.75
% OUTPUT-FLOAT32-NEXT:3: red 300 -3 3000 0.75 | green 400 -4 4000 1.0
% OUTPUT-FLOAT32-NEXT:4: green 400 -4 4000 1.0 | blue 500 -5 5000 1.25
% OUTPUT-FLOAT32-NOT:{{.}}
:- module host_process_flat_values.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, int8, list, process, string, uint16.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type wrapped
    --->    wrapped(int).

    % A record of one functor of numbers, an enumeration and a notag,
    % which Merkurius holds by value.
:- type fdata
    --->    fdata(colour, wrapped, int8, uint16, @T@).

:- func make(int) = fdata.
:- func use(fdata) = string.

make(N) = fdata(
    ( if N mod 3 = 0 then red else if N mod 3 = 1 then green else blue ),
    wrapped(N * 100), int8.det_from_int(-N), uint16.det_from_int(N * 1000),
    @FROM_INT@(N) * 0.25@S@).

use(fdata(C, wrapped(W), I, U, F)) =
    string.string(C) ++ " " ++ string.int_to_string(W) ++ " " ++
    string.int_to_string(int8.to_int(I)) ++ " " ++
    string.int_to_string(uint16.to_int(U)) ++ " " ++
    string.@T@_to_string(F).

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
            process.send(Replies, use(make(K)), !IO),
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
