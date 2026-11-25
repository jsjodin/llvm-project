% MANUAL: Tokens
% Integer literals: decimal, binary, octal, hexadecimal and character
% codes, with underscores and with each size suffix.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[10, 10, 10, 10, 97, 1000000, 65535, 2, 10]
% OUTPUT-NEXT:{10i8, 10i16, 10i32, 10i64}
% OUTPUT-NEXT:{10u, 10u8, 10u16, 10u32, 10u64, 255u8}
% OUTPUT-NOT:{{.}}

:- module run_integers.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

main(!IO) :-
    io.write([10, 0b1010, 0o12, 0xa, 0'a, 1_000_000, 0x_ff_ff, 0b_1_0, 10_i], !IO), io.nl(!IO),
    io.write({10i8, 10i16, 10i32, 10i64}, !IO), io.nl(!IO),
    io.write({10u, 10u8, 10u16, 10u32, 10u64, 0xff_u8}, !IO), io.nl(!IO).
