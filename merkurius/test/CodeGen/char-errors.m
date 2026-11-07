% The char module's det_ functions, and those checking a base, throw the
% library's exceptions, with mmc's messages: det_base_digit_to_int with a
% bad base reports as base_digit_to_int, and det_from_int as the
% predicate. Nothing catches them yet, so the program writes the exception
% to the standard error and exits with status 1. which/1 picks the case.
% The output and messages are mmc's.
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
% RUN: sed 's/^which(0)\./which(4)./' %s > %t.4.m && merkc %t.4.m -o %t.4
% RUN: not %t.4 2> %t.4.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-4 --match-full-lines --strict-whitespace < %t.4.err
% RUN: sed 's/^which(0)\./which(5)./' %s > %t.5.m && merkc %t.5.m -o %t.5
% RUN: not %t.5 2> %t.5.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-5 --match-full-lines --strict-whitespace < %t.5.err
% RUN: sed 's/^which(0)\./which(6)./' %s > %t.6.m && merkc %t.6.m -o %t.6
% RUN: not %t.6 2> %t.6.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-6 --match-full-lines --strict-whitespace < %t.6.err
%
% OUTPUT-0:before
% OUTPUT-0-NEXT:651535Z
% OUTPUT-0-NEXT:after
% OUTPUT:before
% ERROR-1:Uncaught Mercury exception:
% ERROR-1-NEXT:Software Error: predicate `char.det_from_int'/2: Unexpected: conversion failed
% ERROR-2:Uncaught Mercury exception:
% ERROR-2-NEXT:Software Error: function `char.det_hex_digit_to_int'/1: char.hex_digit_to_int failed
% ERROR-3:Uncaught Mercury exception:
% ERROR-3-NEXT:Software Error: predicate `char.base_digit_to_int'/3: base is not in the range 2 .. 36
% ERROR-4:Uncaught Mercury exception:
% ERROR-4-NEXT:Software Error: predicate `char.is_base_digit'/2: invalid base
% ERROR-5:Uncaught Mercury exception:
% ERROR-5-NEXT:Software Error: function `char.det_base_int_to_digit'/2: char.base_int_to_digit failed
% ERROR-6:Uncaught Mercury exception:
% ERROR-6-NEXT:Software Error: predicate `char.base_int_to_digit'/3: invalid base

:- module char_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char.

:- pred which(int::out) is det.
which(0).

main(!IO) :-
    io.write_string("before\n", !IO),
    which(N),
    run(N, !IO).

:- pred run(int::in, io::di, io::uo) is det.
run(N, !IO) :-
    ( if N = 0 then
        io.write_int(char.to_int(char.det_from_int(65)), !IO),
        io.write_int(char.det_hex_digit_to_int('f'), !IO),
        io.write_int(char.det_base_digit_to_int(36, 'Z'), !IO),
        io.write_char(char.det_base_int_to_digit(36, 35), !IO),
        io.write_string("\nafter\n", !IO)
    else if N = 1 then
        io.write_int(char.to_int(char.det_from_int(-1)), !IO)
    else if N = 2 then
        io.write_int(char.det_hex_digit_to_int('g'), !IO)
    else if N = 3 then
        io.write_int(char.det_base_digit_to_int(1, '0'), !IO)
    else if N = 4 then
        ( if char.is_base_digit(37, '0') then
            io.write_string("yes\n", !IO)
        else
            io.write_string("no\n", !IO)
        )
    else if N = 5 then
        io.write_char(char.det_base_int_to_digit(16, 16), !IO)
    else
        ( if char.base_int_to_digit(1, 0, C) then
            io.write_char(C, !IO)
        else
            io.write_string("no\n", !IO)
        )
    ).
