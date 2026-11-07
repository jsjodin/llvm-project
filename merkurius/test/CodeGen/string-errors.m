% The string module's det_ functions throw the library's exceptions, with
% mmc's messages: det_to_int reports as det_base_string_to_int, as in mmc.
% Nothing catches them yet, so the program writes the exception to the
% standard error and exits with status 1. which/1 picks the case. The
% output and messages are mmc's.
%
% RUN: merkc %s -o %t.0 && %t.0 | FileCheck %s --check-prefix=OUTPUT-0 --match-full-lines --strict-whitespace
% RUN: sed 's/^which(0)\./which(1)./' %s > %t.1.m && merkc %t.1.m -o %t.1
% RUN: not %t.1 2> %t.1.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-1 --match-full-lines --strict-whitespace < %t.1.err
% RUN: sed 's/^which(0)\./which(2)./' %s > %t.2.m && merkc %t.2.m -o %t.2
% RUN: not %t.2 2> %t.2.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-2 --match-full-lines --strict-whitespace < %t.2.err
% RUN: sed 's/^which(0)\./which(3)./' %s > %t.3.m && merkc %t.3.m -o %t.3
% RUN: not %t.3 2> %t.3.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-3 --match-full-lines --strict-whitespace < %t.3.err

:- module string_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module string.

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
        io.write_string("\nafter\n", !IO)
    else if N = 1 then
        io.write_char(string.det_index("abc", 3), !IO)
    else if N = 2 then
        io.write_int(string.det_to_int("12x"), !IO)
    else
        io.write_float(string.det_to_float(""), !IO)
    ).

% OUTPUT-0:before
% OUTPUT-0-NEXT:b-122.5
% OUTPUT-0-NEXT:after
% OUTPUT:before
% OUTPUT-NOT:{{.}}
% ERROR-1:Uncaught Mercury exception:
% ERROR-1-NEXT:Software Error: predicate `string.det_index'/3: Unexpected: index out of range
% ERROR-2:Uncaught Mercury exception:
% ERROR-2-NEXT:Software Error: function `string.det_base_string_to_int'/2: Unexpected: conversion failed
% ERROR-3:Uncaught Mercury exception:
% ERROR-3-NEXT:Software Error: function `string.det_to_float'/1: Unexpected: conversion failed.
