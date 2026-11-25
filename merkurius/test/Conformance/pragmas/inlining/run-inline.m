% MANUAL: Inlining
% inline and no_inline pragmas, which do not change what a program does.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:13
% OUTPUT-NOT:{{.}}

:- module run_inline.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pragma inline(double/1).
:- func double(int) = int.
double(X) = 2 * X.
:- pragma no_inline(triple/1).
:- func triple(int) = int.
triple(X) = 3 * X.
:- pragma inline(add/3).
:- pred add(int::in, int::in, int::out) is det.
add(X, Y, X + Y).

main(!IO) :-
    add(double(2), triple(3), Z),
    io.write_int(Z, !IO), io.nl(!IO).
