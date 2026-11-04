% The heap (design document, "Memory management"): cells and arrays are
% allocated by bumping the thread's heap pointer, and when a chunk is full
% the runtime's merk_alloc_slow takes another, so a program allocates far
% more than one chunk; an allocation too large to share a chunk, such as a
% large array's, has a block of its own, between cells of the chunk. A
% ground term known at compile time is a static cell, never allocated, and
% equal ones are one. The output is mmc's, and the same with chunks of 64
% bytes, where nearly every allocation takes the slow path, and every cell
% of more than 32 bytes has a block of its own.
%
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_HEAP_CHUNK_BYTES=64 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: not env MERK_HEAP_CHUNK_BYTES=8 %t 2>&1 | FileCheck %s --check-prefix=BAD
% BAD: Mercury runtime: MERK_HEAP_CHUNK_BYTES=8 is not a number of at least 16

:- module heap.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list.

:- type tree
    --->    leaf
    ;       node(tree, int, tree).

:- func upto(int, list(int)) = list(int).
upto(N, L) = ( if N = 0 then L else upto(N - 1, [N | L]) ).

:- func sum(list(int), int) = int.
sum([], S) = S.
sum([X | Xs], S) = sum(Xs, S + X).

:- func build(int) = tree.
build(D) = ( if D = 0 then leaf else node(build(D - 1), D, build(D - 1)) ).

:- func total(tree) = int.
total(leaf) = 0.
total(node(L, X, R)) = total(L) + X + total(R).

:- pred rounds(int::in, int::in, int::out) is det.
rounds(N, S0, S) :-
    ( if N = 0 then
        S = S0
    else
        rounds(N - 1, S0 + sum(upto(10000, []), 0), S)
    ).

:- func small = list(int).
small = [1, 2, 3].
% LLVM-DAG: @[[THREE:merk.term.[0-9]+]] = private unnamed_addr constant { i64, ptr } { i64 3, ptr null }, align 8
% LLVM-DAG: @[[TWO:merk.term.[0-9]+]] = private unnamed_addr constant { i64, ptr } { i64 2, ptr getelementptr inbounds nuw (i8, ptr @[[THREE]], i64 1) }, align 8
% LLVM-DAG: @[[ONE:merk.term.[0-9]+]] = private unnamed_addr constant { i64, ptr } { i64 1, ptr getelementptr inbounds nuw (i8, ptr @[[TWO]], i64 1) }, align 8
% LLVM-DAG: @[[SHAPE:merk.term.[0-9]+]] = private unnamed_addr constant { ptr, i64, ptr } { ptr getelementptr inbounds nuw (i8, ptr @{{merk.term.[0-9]+}}, i64 1), i64 2, ptr getelementptr inbounds nuw (i8, ptr @{{merk.term.[0-9]+}}, i64 1) }, align 8
% LLVM-LABEL: define internal tailcc ptr @heap.small_f0_0() {
% LLVM-NEXT: ret ptr getelementptr inbounds nuw (i8, ptr @[[ONE]], i64 1)

:- func same = list(int).
same = [1, 2, 3].
% LLVM-LABEL: define internal tailcc ptr @heap.same_f0_0() {
% LLVM-NEXT: ret ptr getelementptr inbounds nuw (i8, ptr @[[ONE]], i64 1)

:- func shape = tree.
shape = node(node(leaf, 1, leaf), 2, node(leaf, 3, leaf)).
% LLVM-LABEL: define internal tailcc ptr @heap.shape_f0_0() {
% LLVM-NEXT: ret ptr getelementptr inbounds nuw (i8, ptr @[[SHAPE]], i64 1)

main(!IO) :-
    % A list of 16 MB.
    io.write_int(sum(upto(1000000, []), 0), !IO),
    io.nl(!IO),
    io.write_int(total(build(16)), !IO),
    io.nl(!IO),
    % 100 lists of 160 KB.
    rounds(100, 0, S),
    io.write_int(S, !IO),
    io.nl(!IO),
    % 800 KB, a block of its own.
    A = array.init(100000, 3),
    L = upto(10, []),
    B = array.init(5, 7),
    io.write_int(array.lookup(A, 99999) + sum(L, 0) + array.lookup(B, 4), !IO),
    io.nl(!IO),
    io.write_int(sum(small, 0) + sum(same, 0) + total(shape), !IO),
    io.nl(!IO).

% OUTPUT:500000500000
% OUTPUT-NEXT:131054
% OUTPUT-NEXT:5000500000
% OUTPUT-NEXT:65
% OUTPUT-NEXT:18
