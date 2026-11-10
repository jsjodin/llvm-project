% The string module's det_ functions throw the library's exceptions, with
% mmc's messages: det_to_int reports as det_base_string_to_int, as in mmc.
% So do those making a string of a NUL or a surrogate code point: most
% report as from_char_list, and first_char as first_char_str_out, naming
% which it was. No copies of one, or padding not needed, are no error.
% Nothing catches them yet, so the program writes the exception to the
% standard error and exits with status 1. which/1 picks the case. The
% output and messages are mmc's.
%
% RUN: merkc %s -o %t.0 && %t.0 | FileCheck %s --check-prefix=OUTPUT-0 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.0.O2 && %t.0.O2 | FileCheck %s --check-prefix=OUTPUT-0 --match-full-lines --strict-whitespace
% RUN: sed 's/^which(0)\./which(1)./' %s > %t.1.m && merkc %t.1.m -o %t.1
% RUN: not %t.1 2> %t.1.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-1 --match-full-lines --strict-whitespace < %t.1.err
% RUN: sed 's/^which(0)\./which(2)./' %s > %t.2.m && merkc %t.2.m -o %t.2
% RUN: not %t.2 2> %t.2.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-2 --match-full-lines --strict-whitespace < %t.2.err
% RUN: sed 's/^which(0)\./which(3)./' %s > %t.3.m && merkc %t.3.m -o %t.3
% RUN: not %t.3 2> %t.3.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-3 --match-full-lines --strict-whitespace < %t.3.err
% RUN: sed 's/^which(0)\./which(4)./' %s > %t.4.m && merkc %t.4.m -o %t.4
% RUN: not %t.4 2> %t.4.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-4 --match-full-lines --strict-whitespace < %t.4.err
% RUN: sed 's/^which(0)\./which(5)./' %s > %t.5.m && merkc %t.5.m -o %t.5
% RUN: not %t.5 2> %t.5.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-5 --match-full-lines --strict-whitespace < %t.5.err
% RUN: sed 's/^which(0)\./which(6)./' %s > %t.6.m && merkc %t.6.m -o %t.6
% RUN: not %t.6 2> %t.6.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-6 --match-full-lines --strict-whitespace < %t.6.err
% RUN: sed 's/^which(0)\./which(7)./' %s > %t.7.m && merkc %t.7.m -o %t.7
% RUN: not %t.7 2> %t.7.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-7 --match-full-lines --strict-whitespace < %t.7.err
% RUN: sed 's/^which(0)\./which(8)./' %s > %t.8.m && merkc %t.8.m -o %t.8
% RUN: not %t.8 2> %t.8.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-8 --match-full-lines --strict-whitespace < %t.8.err
% RUN: sed 's/^which(0)\./which(9)./' %s > %t.9.m && merkc %t.9.m -o %t.9
% RUN: not %t.9 2> %t.9.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-9 --match-full-lines --strict-whitespace < %t.9.err
% RUN: sed 's/^which(0)\./which(10)./' %s > %t.10.m && merkc %t.10.m -o %t.10
% RUN: not %t.10 2> %t.10.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-10 --match-full-lines --strict-whitespace < %t.10.err

:- module string_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, int, list, string.

:- pred which(int::out) is det.
which(0).

main(!IO) :-
    io.write_string("before\n", !IO),
    which(N),
    run(N, !IO).

:- pred run(int::in, io::di, io::uo) is det.
run(N, !IO) :-
    ( if N = 0 then
        io.write_char(string.det_index("abc", 1), !IO),
        io.write_int(string.det_to_int("-12"), !IO),
        io.write_float(string.det_to_float("2.5"), !IO),
        io.write_int(string.length(string.duplicate_char(nul, 0)), !IO),
        io.write_int(string.length(string.duplicate_char(surrogate, -1)), !IO),
        io.write_string(string.pad_left("abcd", nul, 2), !IO),
        io.write_string("\nafter\n", !IO)
    else if N = 1 then
        io.write_char(string.det_index("abc", 3), !IO)
    else if N = 2 then
        io.write_int(string.det_to_int("12x"), !IO)
    else if N = 3 then
        io.write_float(string.det_to_float(""), !IO)
    else if N = 4 then
        io.write_string(string.from_char_list(['a', nul]), !IO)
    else if N = 5 then
        io.write_string(string.from_char_list([surrogate]), !IO)
    else if N = 6 then
        io.write_string(string.duplicate_char(nul, 3), !IO)
    else if N = 7 then
        io.write_string(string.char_to_string(surrogate), !IO)
    else if N = 8 then
        string.first_char(S, nul, "bc"),
        io.write_string(S, !IO)
    else if N = 9 then
        string.first_char(S, surrogate, "bc"),
        io.write_string(S, !IO)
    else
        io.write_string(string.pad_right("ab", surrogate, 4), !IO)
    ).

:- func nul = char.
nul = char.det_from_int(0).

:- func surrogate = char.
surrogate = char.det_from_int(0xd800).

% OUTPUT-0:before
% OUTPUT-0-NEXT:b-122.500abcd
% OUTPUT-0-NEXT:after
% OUTPUT:before
% OUTPUT-NOT:{{.}}
% ERROR-1:Uncaught Mercury exception:
% ERROR-1-NEXT:Software Error: predicate `string.det_index'/3: Unexpected: index out of range
% ERROR-2:Uncaught Mercury exception:
% ERROR-2-NEXT:Software Error: function `string.det_base_string_to_int'/2: Unexpected: conversion failed
% ERROR-3:Uncaught Mercury exception:
% ERROR-3-NEXT:Software Error: function `string.det_to_float'/1: Unexpected: conversion failed.
% ERROR-4:Uncaught Mercury exception:
% ERROR-4-NEXT:Software Error: predicate `string.from_char_list'/2: Unexpected: null character or surrogate code point in list
% ERROR-5:Uncaught Mercury exception:
% ERROR-5-NEXT:Software Error: predicate `string.from_char_list'/2: Unexpected: null character or surrogate code point in list
% ERROR-6:Uncaught Mercury exception:
% ERROR-6-NEXT:Software Error: predicate `string.from_char_list'/2: Unexpected: null character or surrogate code point in list
% ERROR-7:Uncaught Mercury exception:
% ERROR-7-NEXT:Software Error: predicate `string.from_char_list'/2: Unexpected: null character or surrogate code point in list
% ERROR-8:Uncaught Mercury exception:
% ERROR-8-NEXT:Software Error: predicate `string.first_char_str_out'/3: Unexpected: null character
% ERROR-9:Uncaught Mercury exception:
% ERROR-9-NEXT:Software Error: predicate `string.first_char_str_out'/3: Unexpected: surrogate code point
% ERROR-10:Uncaught Mercury exception:
% ERROR-10-NEXT:Software Error: predicate `string.from_char_list'/2: Unexpected: null character or surrogate code point in list
