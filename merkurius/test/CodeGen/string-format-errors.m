% string.format's errors, one of each kind of message: an unknown
% conversion, a missing input or conversion specifier, extra inputs, an
% input of the wrong type, and a width or precision input that is missing
% or not an int, with ordinals in words up to the tenth. Nothing catches
% them yet, so the program writes the exception to the standard error and
% exits with status 1. which/1 picks the case. The output and messages are
% mmc's.
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
% RUN: sed 's/^which(0)\./which(11)./' %s > %t.11.m && merkc %t.11.m -o %t.11
% RUN: not %t.11 2> %t.11.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-11 --match-full-lines --strict-whitespace < %t.11.err
% RUN: sed 's/^which(0)\./which(12)./' %s > %t.12.m && merkc %t.12.m -o %t.12
% RUN: not %t.12 2> %t.12.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-12 --match-full-lines --strict-whitespace < %t.12.err
% RUN: sed 's/^which(0)\./which(13)./' %s > %t.13.m && merkc %t.13.m -o %t.13
% RUN: not %t.13 2> %t.13.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-13 --match-full-lines --strict-whitespace < %t.13.err
% RUN: sed 's/^which(0)\./which(14)./' %s > %t.14.m && merkc %t.14.m -o %t.14
% RUN: not %t.14 2> %t.14.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-14 --match-full-lines --strict-whitespace < %t.14.err
% RUN: sed 's/^which(0)\./which(15)./' %s > %t.15.m && merkc %t.15.m -o %t.15
% RUN: not %t.15 2> %t.15.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-15 --match-full-lines --strict-whitespace < %t.15.err

:- module string_format_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, string.

:- pred which(int::out) is det.
which(0).

main(!IO) :-
    io.write_string("before\n", !IO),
    which(N),
    run(N, !IO).

:- pred run(int::in, io::di, io::uo) is det.
run(N, !IO) :-
    ( if N = 0 then
        io.write_string(string.format("%d %s", [i(1), s("x")]), !IO),
        io.write_string("\nafter\n", !IO)
    else if N = 1 then
        io.write_string(string.format("%hd", [i(1)]), !IO)
    else if N = 2 then
        io.write_string(string.format("%d %d", [i(1)]), !IO)
    else if N = 3 then
        io.write_string(string.format("x", [i(1), s("a")]), !IO)
    else if N = 4 then
        io.write_string(string.format("%d", [i(1), i(2)]), !IO)
    else if N = 5 then
        io.write_string(string.format("%d %d", [i(1), i(2), i(3), i(4)]), !IO)
    else if N = 6 then
        io.write_string(string.format("%", []), !IO)
    else if N = 7 then
        io.write_string(string.format("ab%", [i(1)]), !IO)
    else if N = 8 then
        io.write_string(string.format("%", [i(1), i(2)]), !IO)
    else if N = 9 then
        io.write_string(string.format("%i", [u8(1u8)]), !IO)
    else if N = 10 then
        io.write_string(string.format("%d", [f(1.0)]), !IO)
    else if N = 11 then
        io.write_string(string.format("%c", [s("a")]), !IO)
    else if N = 12 then
        io.write_string(string.format("%s", [c('a')]), !IO)
    else if N = 13 then
        io.write_string(string.format("%*d", [s("a"), i(1)]), !IO)
    else if N = 14 then
        io.write_string(string.format("%.*d", []), !IO)
    else
        io.write_string(string.format("%d %d %d %d %d %d %d %d %d %d %s", [i(1), i(1), i(1), i(1), i(1), i(1), i(1), i(1), i(1), i(1), i(1)]), !IO)
    ).

% OUTPUT-0:before
% OUTPUT-0-NEXT:1 x
% OUTPUT-0-NEXT:after
% OUTPUT:before
% OUTPUT-NOT:{{.}}
% ERROR-1:Uncaught Mercury exception:
% ERROR-1-NEXT:Software Error: string.format: The first conversion specifier uses the unknown specifier character `h'.
% ERROR-2:Uncaught Mercury exception:
% ERROR-2-NEXT:Software Error: string.format: The second conversion specifier, which uses specifier character `d', is missing its input.
% ERROR-3:Uncaught Mercury exception:
% ERROR-3-NEXT:Software Error: string.format: There is no first conversion specifier, but there are 2 inputs.
% ERROR-4:Uncaught Mercury exception:
% ERROR-4-NEXT:Software Error: string.format: There is no second conversion specifier, but there is an extra input.
% ERROR-5:Uncaught Mercury exception:
% ERROR-5-NEXT:Software Error: string.format: There is no third conversion specifier, but there are 2 extra inputs.
% ERROR-6:Uncaught Mercury exception:
% ERROR-6-NEXT:Software Error: string.format: The first conversion specifier is missing, along with its input.
% ERROR-7:Uncaught Mercury exception:
% ERROR-7-NEXT:Software Error: string.format: The first conversion specifier is missing.
% ERROR-8:Uncaught Mercury exception:
% ERROR-8-NEXT:Software Error: string.format: The first conversion specifier is missing, and there are 1 extra inputs.
% ERROR-9:Uncaught Mercury exception:
% ERROR-9-NEXT:Software Error: string.format: The first conversion specifier uses the specifier character `i', but the corresponding input is an 8-bit unsigned integer. The specifiers applicable to uint8s are %o, %x, %X, %u, and %p.
% ERROR-10:Uncaught Mercury exception:
% ERROR-10-NEXT:Software Error: string.format: The first conversion specifier uses the specifier character `d', but the corresponding input is a float. The specifiers applicable to floats are %f, %e, %E, %g and %G.
% ERROR-11:Uncaught Mercury exception:
% ERROR-11-NEXT:Software Error: string.format: The first conversion specifier uses the specifier character `c', but the corresponding input is a string. The only specifier applicable to strings is %s.
% ERROR-12:Uncaught Mercury exception:
% ERROR-12-NEXT:Software Error: string.format: The first conversion specifier uses the specifier character `s', but the corresponding input is a character. The only specifier applicable to characters is %c.
% ERROR-13:Uncaught Mercury exception:
% ERROR-13-NEXT:Software Error: string.format: The first conversion specifier says the width is a runtime input, but the next input is a string, not an integer.
% ERROR-14:Uncaught Mercury exception:
% ERROR-14-NEXT:Software Error: string.format: The first conversion specifier says the precision is a runtime input, but there is no next input.
% ERROR-15:Uncaught Mercury exception:
% ERROR-15-NEXT:Software Error: string.format: The 11th conversion specifier uses the specifier character `s', but the corresponding input is an integer. The specifiers applicable to ints are %d, %i, %o, %x, %X, %u, and %p.
