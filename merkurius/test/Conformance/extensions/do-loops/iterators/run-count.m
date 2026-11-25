% DESIGN: Loops / Iterators
% count(I, Lo, Hi) counts the iterations from Lo, and the last value is
% unified with Hi, so Hi may be free; it never ends the loop. A loop with
% no iterations gives Hi the value Lo - 1.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/count_translation.m
%
% RUN: merkc -O0 -fdo-loops %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[{1, la}, {2, lb}, {3, lc}]
% OUTPUT-NEXT:3
% OUTPUT-NEXT:9
% OUTPUT-NEXT:ends at 1
% OUTPUT-NOT:{{.}}
:- module run_count.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    ( foreach(X, [la, lb, lc]), count(I, 1, N), foreach(P, Numbered) do P = {I, X} ),
    io.write(Numbered, !IO), io.nl(!IO),
    io.write_int(N, !IO), io.nl(!IO),
    ( foreach(_, [] : list(int)), count(_, 10, M) do true ),
    io.write_int(M, !IO), io.nl(!IO),
    ( if ( foreach(_, [xa, ya]), count(_, 0, 1) do true ) then
        io.write_string("ends at 1\n", !IO)
    else
        io.write_string("does not end at 1\n", !IO)
    ).

:- type letter ---> la ; lb ; lc.
:- type xy ---> xa ; ya.
