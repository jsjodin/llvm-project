% DESIGN: Loops / Iterators
% foreach(X, List) takes the elements of List in order. A list bound
% before the loop is traversed; a free one is built, one element per
% iteration, as mode analysis decides.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/foreach_translation.m
%
% RUN: merkc -O0 -fdo-loops %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[3, 3, 5]
% OUTPUT-NEXT:one
% OUTPUT-NEXT:two
% OUTPUT-NEXT:three
% OUTPUT-NEXT:["1a", "2b"]
% OUTPUT-NOT:{{.}}
:- module run_foreach.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, pair, string.

main(!IO) :-
    Words = ["one", "two", "three"],
    ( foreach(W, Words), foreach(L, Lengths) do L = string.length(W) : int ),
    io.write(Lengths, !IO), io.nl(!IO),
    ( foreach(W, Words) do io.write_string(W, !IO), io.nl(!IO) ),
    ( foreach(K - V, [1 - "a", 2 - "b"]), foreach(S, Joined) do
        S = string.int_to_string(K) ++ V
    ),
    io.write(Joined, !IO), io.nl(!IO).
