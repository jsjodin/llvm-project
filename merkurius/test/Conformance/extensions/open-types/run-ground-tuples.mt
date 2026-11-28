% DESIGN: Open terms (Prolog-style variables) / Types
% DESIGN: Sized floating point types
% A tuple of types that cannot hold an unbound variable, beside an open
% field, is ground too, as an int or a list of ints there is
% (run-ground-parts.m), and so is a tuple of such tuples; such a tuple,
% whole or a part of one, may be passed as in. The translation is the same program with the open
% field ground. It is a template, run for float and float32.
% MERKC-FLAGS: -fopen-types
% TRANSLATION: Inputs/ground_tuples_translation.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 -fopen-types %t.float.d/run_ground_tuples_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types %t.float.d/run_ground_tuples_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:three: 3, 2 weights, 0.5, 7
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 -fopen-types %t.float32.d/run_ground_tuples_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types %t.float32.d/run_ground_tuples_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:three: 3, 2 weights, 0.5, 7
% OUTPUT-FLOAT32-NOT:{{.}}
:- module run_ground_tuples.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, list, open, string.

:- type job
    --->    job(open(list(int)), {int, string}, {list(int), {@T@, int}}).

:- pred describe(job::ia, string::out) is det.

describe(job(_, Named, {Ws, Pair}), Text) :-
    named(Named, N, Name),
    pair(Pair, F, K),
    Text = string.format("%s: %d, %d weights, %.1f, %d",
        [s(Name), i(N), i(list.length(Ws)), f(@TO_FLOAT@(F)), i(K)]).

    % Whole tuples, passed as in.
:- pred named({int, string}::in, int::out, string::out) is det.

named({N, Name}, N, Name).

:- pred pair({@T@, int}::in, @T@::out, int::out) is det.

pair({F, K}, F, K).

:- pred make(open(list(int))::ia, job::oa) is det.

make(Holes, job(Holes, {3, "three"}, {[1, 2], {0.5@S@, 7}})).

main(!IO) :-
    L = [_, _] : open(list(int)),
    make(L, Job),
    describe(Job, Text),
    io.write_string(Text ++ "\n", !IO).
