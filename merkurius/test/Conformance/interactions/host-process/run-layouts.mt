% MANUAL: Discriminated unions
% DESIGN: Processes and mailboxes
% DESIGN: Sized floating point types
% Interaction: a type of more functors than a pointer has tags, in another
% process, which makes and uses the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% TRANSLATION: Inputs/host_process_layouts_translation.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 %t.float.d/host_process_layouts_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/host_process_layouts_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0))
% OUTPUT-FLOAT-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9))
% OUTPUT-FLOAT-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0))
% OUTPUT-FLOAT-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 %t.float32.d/host_process_layouts_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/host_process_layouts_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0))
% OUTPUT-FLOAT32-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9))
% OUTPUT-FLOAT32-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0))
% OUTPUT-FLOAT32-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT32-NOT:{{.}}
:- module host_process_layouts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, char, int, list, process, string.

    % More functors with arguments than a pointer has tags, constants, and
    % arguments of every kind of word.
:- type fdata
    --->    c0
    ;       c1
    ;       one(int)
    ;       two(int, string)
    ;       three(@T@, fdata, int)
    ;       four(char, int, int, int)
    ;       five(string)
    ;       six(int)
    ;       seven(fdata, fdata)
    ;       eight(fdata).

:- func make(int) = fdata.
:- func use(fdata) = string.

make(N) = seven(three(@FROM_INT@(N) / 2.0@S@, one(N), N * 3),
    ( if N mod 2 = 0 then
        eight(c0)
    else if N mod 3 = 0 then
        four('x', N, -N, N * N)
    else
        two(N, "s" ++ string.int_to_string(N))
    )).

use(c0) = "c0".
use(c1) = "c1".
use(one(I)) = "one(" ++ string.int_to_string(I) ++ ")".
use(two(I, S)) = "two(" ++ string.int_to_string(I) ++ ", " ++ S ++ ")".
use(three(F, D, I)) = "three(" ++ string.@T@_to_string(F) ++ ", " ++
    use(D) ++ ", " ++ string.int_to_string(I) ++ ")".
use(four(C, A, B, D)) = "four(" ++ string.char_to_string(C) ++ ", " ++
    string.int_to_string(A + B + D) ++ ")".
use(five(S)) = "five(" ++ S ++ ")".
use(six(I)) = "six(" ++ string.int_to_string(I) ++ ")".
use(seven(A, B)) = "seven(" ++ use(A) ++ ", " ++ use(B) ++ ")".
use(eight(D)) = "eight(" ++ use(D) ++ ")".

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
