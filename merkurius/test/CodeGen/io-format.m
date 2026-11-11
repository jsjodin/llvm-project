% io.format/4, which writes string.format's string (string-format.m tests
% the conversions in full), and io.write_strings/3. An error in the
% arguments is string.format's (string-format-errors.m): here, too few.
%
% RUN: merkc %s -o %t && not %t 2>&1 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && not %t2 2>&1 | FileCheck %s --match-full-lines --strict-whitespace

:- module io_format.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, string.

main(!IO) :-
    io.format("%d and %s\n", [i(42), s("text")], !IO),
    io.format("[%5d|%-5d|%05d]\n", [i(7), i(7), i(7)], !IO),
    io.format("%.3f %e %g\n", [f(3.14159), f(1234.5), f(0.0001)], !IO),
    io.format("%c%c %x %o %%\n", [c('o'), c('k'), i(255), i(8)], !IO),
    io.format("%-8s|%8s|\n", [s("left"), s("right")], !IO),
    io.format("no conversions\n", [], !IO),
    io.write_strings(["one", " ", "two", "\n"], !IO),
    io.write_strings([], !IO),
    io.format("%d %d\n", [i(1)], !IO).
% CHECK:42 and text
% CHECK-NEXT:[    7|7    |00007]
% CHECK-NEXT:3.142 1.234500e+03 0.0001
% CHECK-NEXT:ok ff 10 %
% CHECK-NEXT:left    |   right|
% CHECK-NEXT:no conversions
% CHECK-NEXT:one two
% CHECK-NEXT:Uncaught Mercury exception:
% CHECK-NEXT:Software Error: string.format: The second conversion specifier, which uses specifier character `d', is missing its input.
