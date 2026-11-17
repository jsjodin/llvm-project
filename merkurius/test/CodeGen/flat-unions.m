% Flat unions by value (docs/INPROGRESS.md, Flat records, flat unions): a
% type of several constructors whose fields are flat, at most 8 of them in
% all, is an LLVM struct of its constructor's number and each constructor's
% fields, passed and returned by value; cross and step take and make no
% cell, and cross's, too large for registers, is returned in memory. Its
% cell, which the runtime and polymorphic code see, is made where a word or
% another cell wants it: in a list, an array, io.write, a closure, a
% solution, unification and compare through type_infos, list.sort, and
% map_maybe of maybe(int), whose instance is a union too; and the value is
% read back out of it. wide, of 9 fields, is not a union, and stays a cell.
%
% RUN: merkc -fc1 -emit-llvm %s -o %t.ll
% RUN: FileCheck %s --check-prefix=LLVM < %t.ll
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 | FileCheck %s --match-full-lines --strict-whitespace
%
% LLVM-LABEL: define internal tailcc void @flat_unions.cross_f2_0(ptr %0, i64 %1, i64 %2) {
% LLVM-NOT: merk_alloc
% LLVM: store { i64, i64, i64, { i32, i32 }, i32, double } { i64 1, i64 0, i64 0, { i32, i32 } zeroinitializer, i32 0, double 0.000000e+00 }, ptr %{{.*}}, align 8
% LLVM-NOT: merk_alloc
% LLVM: ret void
% LLVM-LABEL: define internal tailcc i64 @flat_unions.step_f2_0(i64 %0, i64 %1) {
% LLVM-NOT: merk_alloc
% LLVM: call tailcc void @flat_unions.cross_f2_0(ptr %{{.*}}, i64 %0, i64 %1)
% LLVM-NOT: merk_alloc
% LLVM: switch i64
% LLVM-NOT: merk_alloc
% LLVM: ret i64
%
% The output is mmc's.
% CHECK:transit(4, 5)
% CHECK-NEXT:escape
% CHECK-NEXT:wall(point(5i32, -1i32), red, celsius(12.5))
% CHECK-NEXT:7
% CHECK-NEXT:[transit(1, 2), wall(point(2i32, -1i32), green, celsius(2.0)), escape, transit(4, 5), wall(point(5i32, -1i32), red, celsius(12.5))]
% CHECK-NEXT:5
% CHECK-NEXT:escape
% CHECK-NEXT:transit(7, 8)
% CHECK-NEXT:equal
% CHECK-NEXT:differ
% CHECK-NEXT:'>'
% CHECK-NEXT:[transit(1, 2), transit(4, 5), escape, wall(point(2i32, 0i32), green, celsius(2.0))]
% CHECK-NEXT:{{\[\[}}transit(3, 4), transit(4, 3)], [transit(4, 3), transit(3, 4)]]
% CHECK-NEXT:110
% CHECK-NEXT:w9(1, 2, 3, 4, 5, 6, 7, 8, 9)
% CHECK-NEXT:yes(6)
% CHECK-NEXT:yes(12)
% CHECK-NEXT:transit(7, 8)

:- module flat_unions.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, int32, list, maybe, solutions, float.

:- type colour ---> red ; green ; blue.
:- type celsius ---> celsius(float).
:- type point ---> point(x :: int32, y :: int32).
:- type crossing
    --->    transit(int, int)
    ;       escape
    ;       wall(point, colour, celsius).
:- type wide
    --->    w0
    ;       w9(int, int, int, int, int, int, int, int, int).

:- func cross(int, int) = crossing.
cross(A, B) = C :-
    ( if A mod 3 = 0 then
        C = escape
    else if A mod 3 = 1 then
        C = transit(A, B)
    else
        C = wall(point(int32.cast_from_int(A), int32.cast_from_int(A - B)),
            ( if A > 3 then red else green ), celsius(float(A * A) / 2.0))
    ).

    % What a walk does at a crossing, as qs_track's caller of crossing.
:- func step(int, int) = int.
step(A, B) = N :-
    C = cross(A, B),
    (
        C = transit(X, Y),
        N = X + Y
    ;
        C = escape,
        N = -1
    ;
        C = wall(P, H, _),
        N = int32.to_int(P ^ x) + ( H = red -> 100 ; 0 )
    ).

    % A union accumulated through a loop.
:- pred last_transit(int::in, int::in, crossing::in, crossing::out) is det.
last_transit(I, N, !C) :-
    ( if I > N then
        true
    else
        C1 = cross(I, I + 1),
        ( if C1 = transit(_, _) then !:C = C1 else true ),
        last_transit(I + 1, N, !C)
    ).

:- pred pair(crossing::out, crossing::out) is multi.
pair(transit(3, 4), transit(4, 3)).
pair(transit(4, 3), transit(3, 4)).

main(!IO) :-
    io.write_line(cross(4, 5), !IO),
    io.write_line(cross(3, 5), !IO),
    io.write_line(cross(5, 6), !IO),
    io.write_int(list.foldl(func(I, S) = S + step(I, 1), 1 .. 3, 0) + 4, !IO),
    io.nl(!IO),
    L = list.map(func(I) = cross(I, I + 1), 1 .. 5),
    io.write_line(L, !IO),
    io.write_int(list.length(L), !IO), io.nl(!IO),
    A = array(L),
    io.write_line(A ^ elem(2), !IO),
    A1 = A ^ elem(2) := transit(7, 8),
    io.write_line(A1 ^ elem(2), !IO),
    ( if cross(4, 5) = transit(4, 5) then io.write_string("equal\n", !IO) else io.write_string("differ\n", !IO) ),
    ( if cross(4, 5) = cross(3, 5) then io.write_string("equal\n", !IO) else io.write_string("differ\n", !IO) ),
    compare(R, cross(3, 5), cross(4, 5)), io.write_line(R, !IO),
    io.write_line(list.sort([cross(4, 5), cross(2, 2), cross(3, 0), cross(1, 2)])
        `with_type` list(crossing), !IO),
    solutions(
        ( pred(E::out) is multi :-
            pair(U, V),
            E = [U, V]
        ), Pairs),
    io.write_line(Pairs, !IO),
    io.write_line(step(5, 1) + step(4, 1), !IO),
    W = ( if list.length(L) `with_type` int > 3 then w9(1, 2, 3, 4, 5, 6, 7, 8, 9) else w0 ),
    io.write_line(W, !IO),
    M = ( if list.length(L) `with_type` int > 3 then yes(6) else no ),
    io.write_line(M, !IO),
    io.write_line(map_maybe(func(X) = X * 2, M), !IO),
    last_transit(1, 9, escape, Last),
    io.write_line(Last, !IO).
