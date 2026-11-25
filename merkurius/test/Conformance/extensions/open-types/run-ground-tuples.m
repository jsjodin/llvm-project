% DESIGN: Open terms (Prolog-style variables) / Types
% A tuple of types that cannot hold an unbound variable, beside an open
% field, is ground too, as an int or a list of ints there is
% (run-ground-parts.m), and so is a tuple of such tuples; such a tuple,
% whole or a part of one, may be passed as in. The translation is the same program with the open
% field ground.
% MERKC-FLAGS: -fopen-types
% TRANSLATION: Inputs/ground_tuples_translation.m
%
% RUN: merkc -O0 -fopen-types %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:three: 3, 2 weights, 0.5, 7
% OUTPUT-NOT:{{.}}
:- module run_ground_tuples.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open, string.

:- type job
    --->    job(open(list(int)), {int, string}, {list(int), {float, int}}).

:- pred describe(job::ia, string::out) is det.

describe(job(_, Named, {Ws, Pair}), Text) :-
    named(Named, N, Name),
    pair(Pair, F, K),
    Text = string.format("%s: %d, %d weights, %.1f, %d",
        [s(Name), i(N), i(list.length(Ws)), f(F), i(K)]).

    % Whole tuples, passed as in.
:- pred named({int, string}::in, int::out, string::out) is det.

named({N, Name}, N, Name).

:- pred pair({float, int}::in, float::out, int::out) is det.

pair({F, K}, F, K).

:- pred make(open(list(int))::ia, job::oa) is det.

make(Holes, job(Holes, {3, "three"}, {[1, 2], {0.5, 7}})).

main(!IO) :-
    L = [_, _] : open(list(int)),
    make(L, Job),
    describe(Job, Text),
    io.write_string(Text ++ "\n", !IO).
