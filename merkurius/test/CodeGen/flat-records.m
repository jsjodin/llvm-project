% Flat records by value (docs/INPROGRESS.md, Flat records, step 1): a type of
% one constructor of numbers, characters, enumerations, notags of those and
% such records, and a tuple of them, is an LLVM struct, a nested record
% inline, passed and returned by value; tile's procedures take no cell. Its
% cell, which the runtime and polymorphic code see, is made where a word or
% another cell wants it: in a list, an array, io.write, a closure's foldl,
% unification and compare through type_infos, and list.sort; and the record
% is read back out of it. bump's output, too large for registers, is
% returned in memory.
%
% RUN: merkc -fc1 -emit-llvm %s -o %t.ll
% RUN: FileCheck %s --check-prefix=LLVM < %t.ll
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 | FileCheck %s --match-full-lines --strict-whitespace
%
% LLVM-LABEL: define internal tailcc i64 @flat_records.score_f1_0({ { i32, i32 }, i32, i8, i32, double, double } %0) {
% LLVM-NOT: merk_alloc
% LLVM: extractvalue { i32, i32 }
% LLVM-NOT: merk_alloc
% LLVM: ret i64
% LLVM-LABEL: define internal tailcc void @flat_records.bump_2_0(ptr %0, { { i32, i32 }, i32, i8, i32, double, double } %1) {
% LLVM-NOT: merk_alloc
% LLVM: store { { i32, i32 }, i32, i8, i32, double, double } %{{.*}}, ptr %{{.*}}, align 8
% LLVM-NEXT: ret void
%
% The output is mmc's.
% CHECK:tile(point(3i32, 6i32), 9i32, 111u8, red, celsius(1.5), 4.5)
% CHECK-NEXT:129
% CHECK-NEXT:10i32
% CHECK-NEXT:point(3i32, 6i32)
% CHECK-NEXT:[tile(point(1i32, 2i32), 1i32, 37u8, green, celsius(0.5), 1.5), tile(point(2i32, 4i32), 4i32, 74u8, blue, celsius(1.0), 3.0), tile(point(3i32, 6i32), 9i32, 111u8, red, celsius(1.5), 4.5), tile(point(4i32, 8i32), 16i32, 148u8, green, celsius(2.0), 6.0)]
% CHECK-NEXT:1430
% CHECK-NEXT:tile(point(3i32, 6i32), 9i32, 111u8, red, celsius(1.5), 4.5)
% CHECK-NEXT:red
% CHECK-NEXT:equal
% CHECK-NEXT:differ
% CHECK-NEXT:'<'
% CHECK-NEXT:2i32
% CHECK-NEXT:[tile(point(2i32, 4i32), 4i32, 74u8, blue, celsius(1.0), 3.0), tile(point(4i32, 8i32), 16i32, 148u8, green, celsius(2.0), 6.0), tile(point(5i32, 10i32), 25i32, 185u8, blue, celsius(2.5), 7.5)]
% CHECK-NEXT:{{\[\[}}2u8, 111u8, 20u8], [111u8, 1u8, 129u8]]

:- module flat_records.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, int32, list, solutions, uint8, float, string.

:- type colour ---> red ; green ; blue.
:- type celsius ---> celsius(float).
:- type point ---> point(x :: int32, y :: int32).
:- type tile
    --->    tile(
                pos :: point,
                live :: int32,
                edges :: uint8,
                hue :: colour,
                temp :: celsius,
                weight :: float
            ).

:- func make(int) = tile.
make(I) = tile(point(int32.cast_from_int(I), int32.cast_from_int(I * 2)),
    int32.cast_from_int(I * I), uint8.cast_from_int(I * 37), colour_of(I),
    celsius(float(I) / 2.0), float(I) * 1.5).

:- func colour_of(int) = colour.
colour_of(I) = ( if I mod 3 = 0 then red else if I mod 3 = 1 then green else blue ).

:- func score(tile) = int.
score(tile(point(X, Y), L, E, H, _, _)) =
    int32.to_int(X) + int32.to_int(Y) + int32.to_int(L) + uint8.to_int(E) +
    ( H = blue -> 1000 ; 0 ).

:- pred bump(tile::in, tile::out) is det.
bump(T0, T) :-
    T = T0 ^ live := T0 ^ live + 1i32.

% Each clause tail-calls the continuation with two records, more stack
% arguments than it received, which reserves room for them below its return
% address; a collection in the continuation walks through its frame
% (runtime/frame-info.h). The records live in the continuation's
% environment, too.
:- pred pick(tile::in, tile::out, tile::out) is multi.
pick(T, T, T ^ edges := 1u8).
pick(T, T ^ edges := 2u8, T).

main(!IO) :-
    T1 = make(3),
    io.write_line(T1, !IO),
    io.write_int(score(T1), !IO), io.nl(!IO),
    bump(T1, T2),
    io.write_line(T2 ^ live, !IO),
    io.write_line(T2 ^ pos, !IO),
    L = list.map(make, 1 .. 4),
    io.write_line(L, !IO),
    io.write_line(list.foldl(func(T, Sum) = Sum + score(T), L, 0), !IO),
    A = array(L),
    io.write_line(A ^ elem(2), !IO),
    A1 = A ^ elem(1) := T2,
    io.write_line(A1 ^ elem(1) ^ hue, !IO),
    ( if T1 = make(3) then io.write_string("equal\n", !IO) else io.write_string("differ\n", !IO) ),
    ( if T1 = T2 then io.write_string("equal\n", !IO) else io.write_string("differ\n", !IO) ),
    compare(R, T1, T2), io.write_line(R, !IO),
    ( if make(2) = tile(point(P, _), _, _, _, _, _) then io.write_line(P, !IO) else true ),
    io.write_line(list.sort([make(5), make(2), make(4)]) `with_type` list(tile), !IO),
    solutions(
        ( pred(E::out) is multi :-
            pick(T1, U, V),
            E = [U ^ edges, V ^ edges, uint8.cast_from_int(score(U))]
        ), Edges),
    io.write_line(Edges, !IO).
