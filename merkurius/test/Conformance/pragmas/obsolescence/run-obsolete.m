% MANUAL: Obsolescence
% Calls of obsolete predicates, functions and procedures, which merkc
% warns of as mmc does (Semantics/pragmas/obsolete.m), but which run.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:2
% OUTPUT-NOT:{{.}}

:- module run_obsolete.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pragma obsolete(old/1).
:- func old(int) = int.
old(X) = X + 1.
:- pragma obsolete(old2/2, [new/2, newer/2]).
:- pred old2(int::in, int::out) is det.
old2(X, X).
:- pred new(int::in, int::out) is det.
new(X, X).
:- pred newer(int::in, int::out) is det.
newer(X, X).
:- pred m(int, int).
:- mode m(in, out) is det.
:- mode m(out, in) is det.
:- pragma obsolete_proc(m(out, in), [new/2]).
m(X, X).
:- func f(int) = int.
:- mode f(in) = out is det.
:- pragma obsolete_proc(f(in) = out).
f(X) = X.

main(!IO) :-
    A = old(1), old2(A, B), m(B, C), m(D, C), E = f(D),
    io.write_int(E, !IO), io.nl(!IO).
