% MANUAL: Tokens
% Floating point literals with a fraction, an exponent or both, a signed
% exponent, and underscores between digits.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[1.5, 1000.0, 150.0, 0.25, 100.0, 1000.0001, 100000000000.0]
% OUTPUT-NOT:{{.}}

:- module run_floats.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

main(!IO) :-
    io.write([1.5, 1e3, 1.5E2, 2.5e-1, 1e+2, 1_000.000_1, 1_0e1_0], !IO), io.nl(!IO).
