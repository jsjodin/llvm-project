% MANUAL: Discriminated unions
% Constructors may be overloaded among types: x/0 and y/1 belong to two
% types each, and the types around them choose which.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:a.x
% OUTPUT-NEXT:1
% OUTPUT-NEXT:b.x
% OUTPUT-NEXT:"s"
% OUTPUT-NEXT:b.z
% OUTPUT-NOT:{{.}}

:- module run_overloaded_constructors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type a
    --->    x
    ;       y(int).

:- type b
    --->    x
    ;       y(string)
    ;       z.

main(!IO) :-
    write_a(x, !IO),
    write_a(y(1), !IO),
    write_b(x, !IO),
    write_b(y("s"), !IO),
    write_b(z, !IO).

:- pred write_a(a::in, io::di, io::uo) is det.

write_a(A, !IO) :-
    ( A = x, io.write_string("a.x\n", !IO)
    ; A = y(N), io.write_line(N, !IO)
    ).

:- pred write_b(b::in, io::di, io::uo) is det.

write_b(B, !IO) :-
    ( B = x, io.write_string("b.x\n", !IO)
    ; B = y(S), io.write_line(S, !IO)
    ; B = z, io.write_string("b.z\n", !IO)
    ).
