% DESIGN: Open terms (Prolog-style variables) / Types
% Values of an open type have inst any, Mercury's inst for values that may
% be partly unbound. A part of such a term whose type cannot hold an
% unbound variable, an int, a string or a list of ints beside an open
% field, is ground, as a part of a solver term is, and may be passed as
% in.
% MERKC-FLAGS: -fopen-types
% TRANSLATION: Inputs/ground_parts_translation.pl
%
% RUN: merkc -O0 -fopen-types %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:three: 3, 3 weights summing to 6
% OUTPUT-NOT:{{.}}
:- module run_ground_parts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open, string.

:- type job
    --->    job(open(list(int)), int, string, list(int)).

:- pred describe(job::ia, string::out) is det.

describe(job(_, N, Name, Weights), Text) :-
    Text = string.format("%s: %d, %d weights summing to %d",
        [s(Name), i(N), i(list.length(Weights)),
        i(list.foldl(int.plus, Weights, 0))]).

:- pred make(open(list(int))::ia, job::oa) is det.

make(Holes, job(Holes, 3, "three", [1, 2, 3])).

main(!IO) :-
    L = [_, _] : open(list(int)),
    make(L, Job),
    describe(Job, Text),
    io.write_string(Text ++ "\n", !IO).
