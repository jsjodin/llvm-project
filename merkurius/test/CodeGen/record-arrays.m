% Arrays of flat records hold them inline, as C's arrays of structs do
% (runtime/array.h, width code 4): a tile is { { i32, i32 }, i32, i8, i32,
% double }, 32 bytes with an enumeration in 4 and C's padding, after the
% header's two words, the second the record's size. Monomorphic code reads
% and writes a record in place, and a loop reads a field alone; polymorphic
% code, array.m's and the program's own, boxes a record as it is read and
% unboxes it as it is stored, by the element's type_info. Each array
% procedure that makes, copies, fills, resizes or walks an array is used on
% records, as are unification, comparison and io.write, a record with a
% notag and a character, a tuple of numbers and an empty array. The output
% is mmc's, and the collector runs at every allocation, checking its roots.
%
% RUN: merkc -fc1 -emit-llvm %s -o - | FileCheck %s --check-prefix=O0
% RUN: merkc -fc1 -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=O2
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t0 2>/dev/null | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 2>/dev/null | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% An element is read whole, from after the header.
% O0-LABEL: define {{.*}} @record_arrays.live_sum_f3_0(
% O0: %[[AT:.*]] = getelementptr i64, ptr addrspace(1) %0, i32 2
% O0: %[[TILE:.*]] = getelementptr { { i32, i32 }, i32, i8, i32, double }, ptr addrspace(1) %[[AT]], i64 %1
% O0: load { { i32, i32 }, i32, i8, i32, double }, ptr addrspace(1) %[[TILE]]

% Optimized, the loop reads the live field alone, 24 bytes into the array's
% 32-byte step: 16 of the header and 8 of the point.
% O2: call ptr addrspace(1) @merk_array_new_bytes(i64 4, ptr {{.*}}, i64 32)
% O2: getelementptr [32 x i8], ptr addrspace(1) %{{.*}}, i64 %{{.*}}
% O2-NEXT: %[[LIVE:.*]] = getelementptr i8, ptr addrspace(1) %{{.*}}, i64 24
% O2-NEXT: load i32, ptr addrspace(1) %[[LIVE]], align 4

% OUTPUT:array([tile(point(0i32, 0i32), 0i32, 0u8, red, 0.0), tile(point(1i32, -1i32), 1i32, 37u8, green, 0.25), tile(point(2i32, -2i32), 4i32, 74u8, blue, 0.5), tile(point(3i32, -3i32), 9i32, 111u8, red, 0.75)])
% OUTPUT-NEXT:14
% OUTPUT-NEXT:tile(point(1i32, -1i32), 1i32, 37u8, green, 0.25)
% OUTPUT-NEXT:tile(point(5i32, -5i32), 25i32, 185u8, blue, 1.25)
% OUTPUT-NEXT:4
% OUTPUT-NEXT:array([tile(point(1i32, -1i32), 1i32, 37u8, green, 0.25), tile(point(2i32, -2i32), 4i32, 74u8, blue, 0.5), tile(point(3i32, -3i32), 9i32, 111u8, red, 0.75), tile(point(4i32, -4i32), 16i32, 148u8, green, 1.0), tile(point(5i32, -5i32), 25i32, 185u8, blue, 1.25)])
% OUTPUT-NEXT:tile(point(9i32, -9i32), 81i32, 77u8, red, 2.25)
% OUTPUT-NEXT:array([tile(point(1i32, -1i32), 1i32, 37u8, green, 0.25), tile(point(2i32, -2i32), 4i32, 74u8, blue, 0.5)])
% OUTPUT-NEXT:array([tile(point(1i32, -1i32), 1i32, 37u8, green, 0.25), tile(point(2i32, -2i32), 4i32, 74u8, blue, 0.5)])
% OUTPUT-NEXT:33
% OUTPUT-NEXT:"equal"
% OUTPUT-NEXT:'<'
% OUTPUT-NEXT:array([tile(point(4i32, -4i32), 16i32, 148u8, green, 1.0), tile(point(4i32, -4i32), 16i32, 148u8, green, 1.0)])
% OUTPUT-NEXT:tile(point(3i32, -3i32), 9i32, 111u8, red, 0.75)
% OUTPUT-NEXT:array([label(id(1), 'B', 37u8), label(id(4), 'C', 74u8), label(id(9), 'D', 111u8), label(id(16), 'E', 148u8), label(id(25), 'F', 185u8), label(id(81), 'J', 77u8), label(id(81), 'J', 77u8)])
% OUTPUT-NEXT:array([{1, 2.5}, {3, 4.5}])
% OUTPUT-NEXT:array([point(3i32, 4i32), point(1i32, 2i32)])
% OUTPUT-NEXT:array([])
% OUTPUT-NOT:{{.}}

:- module record_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, char, float, int, int32, list, uint8.

:- type colour
    --->    red
    ;       green
    ;       blue.
:- type point ---> point(x :: int32, y :: int32).
:- type id ---> id(int).
:- type tile
    --->    tile(
                pos :: point,
                live :: int32,
                edges :: uint8,
                hue :: colour,
                heat :: float
            ).
:- type label
    --->    label(
                key :: id,
                letter :: char,
                small :: uint8
            ).

:- func make(int) = tile.
make(I) = tile(point(int32.cast_from_int(I), int32.cast_from_int(-I)),
    int32.cast_from_int(I * I), uint8.cast_from_int(I * 37), colour_of(I),
    float(I) / 4.0).

:- func colour_of(int) = colour.
colour_of(I) =
    ( if I mod 3 = 0 then red else if I mod 3 = 1 then green else blue ).

:- func live_sum(array(tile), int, int) = int.
live_sum(A, I, S) =
    ( if I >= array.size(A) then
        S
    else
        live_sum(A, I + 1, S + int32.to_int(A ^ elem(I) ^ live))
    ).

    % Polymorphic: through array.m's procedures and the program's own.
:- func second(array(T)) = T.
second(A) = A ^ elem(1).

:- pred poke(int::in, T::in, array(T)::array_di, array(T)::array_uo) is det.
poke(I, X, !A) :-
    array.set(I, X, !A).

:- func x_of(tile, int) = int.
x_of(T, S) = S + int32.to_int(T ^ pos ^ x).

:- func to_label(tile) = label.
to_label(T) = label(id(int32.to_int(T ^ live)),
    char.det_from_int(0x41 + int32.to_int(T ^ pos ^ x)), T ^ edges).

:- pred show(T::in, io::di, io::uo) is det.
show(X, !IO) :-
    io.write(X, !IO),
    io.nl(!IO).

main(!IO) :-
    A0 = array.init(4, make(0)),
    array.set(1, make(1), A0, A1),
    array.set(2, make(2), A1, A2),
    A3 = A2 ^ elem(3) := make(3),
    show(A3, !IO),
    show(live_sum(A3, 0, 0), !IO),
    show(second(A3), !IO),
    poke(0, make(5), array.copy(A3), A4),
    show(A4 ^ elem(0), !IO),
    show(list.length(array.to_list(A4)) : int, !IO),
    B = array.from_list(list.map(make, 1 .. 5)),
    show(B, !IO),
    array.resize(7, make(9), array.copy(B), C),
    show(C ^ elem(6), !IO),
    array.shrink(2, array.copy(C), D),
    show(D, !IO),
    array.copy(D, E),
    show(E, !IO),
    show(array.foldl(x_of, C, 0), !IO),
    ( if array.copy(A4, A5), A4 = A5 then
        show("equal", !IO)
    else
        show("differ", !IO)
    ),
    compare(R, B, C),
    show(R, !IO),
    array.fill(make(4), array.copy(D), G),
    show(G, !IO),
    array.swap(0, 2, array.copy(C), H),
    show(H ^ elem(0), !IO),
    show(array.map(to_label, C), !IO),
    T2 = array.from_list([{1, 2.5}, {3, 4.5}]) : array({int, float}),
    show(T2, !IO),
    show(array.from_reverse_list([point(1i32, 2i32), point(3i32, 4i32)])
        : array(point), !IO),
    show(array.init(0, make(1)) : array(tile), !IO).
