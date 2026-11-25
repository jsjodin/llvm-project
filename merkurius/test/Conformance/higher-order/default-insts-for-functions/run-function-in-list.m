% MANUAL: Default insts for functions
% A function term taken out of a polymorphic collection has no
% higher-order inst, so it gets the default func(in) = out is det, and can
% be applied.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:11
% OUTPUT-NEXT:[6, 10]
% OUTPUT-NOT:{{.}}

:- module run_function_in_list.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    Fs = [(func(X) = X + 1), (func(X) = X * 2)],
    F = list.det_head(Fs),
    io.write_line(apply(F, 10), !IO),
    io.write_line(list.map((func(G) = G(5)), Fs), !IO).
