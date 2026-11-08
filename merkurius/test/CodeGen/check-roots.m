% With MERK_CHECK_ROOTS, each allocation's slow path walks the stack, and checks
% each root against its type (docs/INPROGRESS.md, Statepoints and stack maps).
% A small chunk makes most allocations take the slow path.
%
% RUN: merkc -O0 %s -o %t0
% RUN: env MERK_CHECK_ROOTS=1 MERK_HEAP_CHUNK_BYTES=64 %t0 2>&1 >/dev/null | FileCheck %s --check-prefix=ROOTS
% RUN: merkc -O2 %s -o %t2
% RUN: env MERK_CHECK_ROOTS=1 MERK_HEAP_CHUNK_BYTES=64 %t2 2>&1 >/dev/null | FileCheck %s --check-prefix=ROOTS
% RUN: %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% ROOTS: Mercury runtime: roots: {{[1-9][0-9]*}} safepoints, {{[1-9][0-9]*}} frames, {{[0-9]+}} roots, {{[0-9]+}} environments, 0 slow paths not at a safepoint

:- module check_roots.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- type tree ---> leaf ; node(tree, int, tree).

:- pragma no_inline(insert/2).
:- func insert(int, tree) = tree.
insert(X, leaf) = node(leaf, X, leaf).
insert(X, node(L, Y, R)) =
    ( if X < Y then node(insert(X, L), Y, R) else node(L, Y, insert(X, R)) ).

:- func sum(tree) = int.
sum(leaf) = 0.
sum(node(L, X, R)) = sum(L) + X + sum(R).

main(!IO) :-
    T = list.foldl(insert, [5, 3, 8, 1, 4, 7, 9], leaf),
    io.write_int(sum(T), !IO),
    io.nl(!IO).

% OUTPUT:37
