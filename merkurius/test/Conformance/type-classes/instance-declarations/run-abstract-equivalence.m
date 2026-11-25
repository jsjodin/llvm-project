% MANUAL: Instance declarations
% The manual says an instance's type may not be an abstract type that is
% defined elsewhere as an equivalence type, but mmc accepts one, and the
% program runs (design document, "Where mmc and the manual disagree").
% MODULES: Inputs/hidden_int.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/hidden_int.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/hidden_int.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:0
% OUTPUT-NOT:{{.}}

:- module run_abstract_equivalence.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module hidden_int.

:- typeclass c(T) where [
    func f(T) = int
].

:- instance c(hidden) where [
    f(_) = 0
].

main(!IO) :-
    io.write_int(f(make(1)), !IO).
