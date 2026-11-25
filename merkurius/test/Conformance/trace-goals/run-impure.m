% MANUAL: Trace goals
% A trace goal may call an impure predicate: its clause is implicitly
% promised pure.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:tick
% OUTPUT-NEXT:1
% OUTPUT-NOT:{{.}}

:- module run_impure.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pragma foreign_decl("C", "#include <stdio.h>").
:- impure pred tick is det.
:- pragma foreign_proc("C", tick, [will_not_call_mercury], "puts(\"tick\");").
:- pred p(int::out) is det.
p(X) :- trace [] ( impure tick ), X = 1.
main(!IO) :- p(X), io.write_int(X, !IO), io.nl(!IO).
