% MANUAL: Character type
% A char is a Unicode code point: ASCII, a two-byte and a four-byte code
% point convert to and from their numbers, and print as themselves.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:97 a
% OUTPUT-NEXT:233 é
% OUTPUT-NEXT:128512 😀
% OUTPUT-NEXT:7 control
% OUTPUT-NEXT:1114111
% OUTPUT-NOT:{{.}}

:- module run_code_points.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, int, list, string.

main(!IO) :-
    Chars = ['a', '\u00e9', '\U0001F600', '\x7\'],
    list.foldl(write_code, Chars, !IO),
    ( if char.from_int(0x10FFFF, C) then
        io.write_line(char.to_int(C) : int, !IO)
    else
        io.write_string("no U+10FFFF\n", !IO)
    ).

:- pred write_code(char::in, io::di, io::uo) is det.

write_code(C, !IO) :-
    io.write_int(char.to_int(C), !IO),
    io.write_string(" ", !IO),
    ( if char.to_int(C) >= 32 then
        io.write_char(C, !IO)
    else
        io.write_string("control", !IO)
    ),
    io.nl(!IO).
