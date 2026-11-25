% DESIGN: Flat records
% A type of one constructor of numbers, characters, enumerations, notags of
% those and such records is held by value, and an array of them inline,
% as C lays it out. None of that is visible: a program means what it means
% in Mercury, so mmc is the oracle. The program uses records and arrays of
% them where a word or a cell is wanted: in lists, pairs and closures,
% through polymorphic array code (foldl, map, sort, binary search), in
% unification, comparison and io.write, after resizing and copying, and
% with narrow fields of every size.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:particle(point(2i32, 3i32), blue, 'f', -45i8, -815i16, 25u8, celsius(2.5), 6.25, 5000015u64)
% OUTPUT-NEXT:particle(point(0i32, 7i32), blue, 'd', -47i8, -889i16, 15u8, celsius(1.5), 2.25, 3000009u64)
% OUTPUT-NEXT:point(100i32, -100i32)
% OUTPUT-NEXT:{-962, 249762}
% OUTPUT-NEXT:1396
% OUTPUT-NEXT:5135.0
% OUTPUT-NEXT:particle(point(-3i32, 0i32), red, 'a', -50i8, -1000i16, 0u8, celsius(0.0), 0.0, 0u64)
% OUTPUT-NEXT:4000012u64
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:[-962, -904, -858, -812, -718]
% OUTPUT-NEXT:{43, yes, yes}
% OUTPUT-NEXT:array([particle(point(-3i32, 0i32), red, 'a', -50i8, -1000i16, 0u8, celsius(0.0), 0.0, 0u64), particle(point(-2i32, 11i32), green, 'b', -49i8, -963i16, 5u8, celsius(0.5), 0.25, 1000003u64)])
% OUTPUT-NEXT:40
% OUTPUT-NEXT:[point(-3i32, 0i32) - red, point(-3i32, 8i32) - blue, point(-3i32, 9i32) - green]
% OUTPUT-NEXT:{(<), yes, no}
% OUTPUT-NEXT:array([point(1i32, 2i32), point(-3i32, 4i32), point(0i32, 0i32), point(0i32, 0i32)])
% OUTPUT-NEXT:-71
% OUTPUT-NOT:{{.}}
:- module run_record_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, bool, char, float, int.
:- use_module int8, int16, int32, uint8, uint64.
:- import_module list, pair, string.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type celsius
    --->    celsius(float).

:- type point
    --->    point(x :: int32, y :: int32).

:- type particle
    --->    particle(
                at          :: point,
                colour      :: colour,
                mark        :: char,
                small       :: int8,
                mid         :: int16,
                byte        :: uint8,
                temp        :: celsius,
                weight      :: float,
                id          :: uint64
            ).

:- func make(int) = particle.

make(I) = particle(point(int32.det_from_int(I mod 7 - 3),
        int32.det_from_int((I * 11) mod 13)),
    ( if I mod 3 = 0 then red else if I mod 3 = 1 then green else blue ),
    char.det_from_int(char.to_int('a') + I mod 26),
    int8.det_from_int(I mod 100 - 50),
    int16.det_from_int((I * 37) mod 2000 - 1000),
    uint8.det_from_int((I * 5) mod 256), celsius(float(I) * 0.5),
    float(I * I) / 4.0, uint64.det_from_int(I * 1000003)).

:- func same(T, T) = bool.

same(A, B) = ( if A = B then yes else no ).

:- func score(particle) = int.

score(P) = int32.to_int(P ^ at ^ x) * 3 + int32.to_int(P ^ at ^ y) +
    int8.to_int(P ^ small) + int16.to_int(P ^ mid) + uint8.to_int(P ^ byte) +
    char.to_int(P ^ mark) + truncate_to_int(P ^ weight) +
    ( if P ^ temp = celsius(T), T > 10.0 then 1 else 0 ).

:- pred by_score(particle::in, particle::in, comparison_result::out) is det.

by_score(A, B, R) :-
    compare(R0, score(A), score(B)),
    ( if R0 = (=) then compare(R, A ^ id, B ^ id) else R = R0 ).

main(!IO) :-
    N = 40,
    A0 = array.generate(N, make) : array(particle),
    io.write_line(A0 ^ elem(5), !IO),
    % Updating fields of an element in place, and whole elements.
    A1 = (A0 ^ elem(3) ^ colour := blue) : array(particle),
    E4 = (A1 ^ elem(4)) ^ at := point(100i32, -100i32),
    A2 = A1 ^ elem(4) := E4,
    io.write_line(A2 ^ elem(3), !IO),
    io.write_line(A2 ^ elem(4) ^ at, !IO),
    % A copy is independent of the array it copies.
    array.copy(A2, Copy0),
    array.set(0, make(999), Copy0, Copy),
    io.write_line({score(A2 ^ elem(0)), score(Copy ^ elem(0))}, !IO),
    % Polymorphic code over the array: foldl, map, sort, binary search.
    Total = array.foldl((func(P, S) = S + score(P)), A2, 0),
    io.write_line(Total, !IO),
    Weights = array.map((func(P) = P ^ weight), A2),
    io.write_line(array.foldl((func(W, S) = S + W), Weights, 0.0), !IO),
    array.copy(A2, ToSort),
    Sorted = array.sort(ToSort),
    io.write_line(Sorted ^ elem(0), !IO),
    io.write_line(Sorted ^ elem(N - 1) ^ id, !IO),
    ( if array.binary_search(Sorted, A2 ^ elem(17), I) then
        io.write_line(same(Sorted ^ elem(I), A2 ^ elem(17)), !IO)
    else
        io.write_string("not found\n", !IO)
    ),
    list.sort(by_score, array.to_list(A2), ByScore),
    list.take_upto(5, ByScore, Lowest),
    io.write_line(list.map(score, Lowest) : list(int), !IO),
    % Resizing keeps the elements and fills the rest.
    array.copy(A2, ToResize),
    array.resize(N + 3, make(0), ToResize, Bigger),
    Size = array.size(Bigger) : int,
    io.write_line({Size, same(Bigger ^ elem(N + 2), make(0)),
        same(Bigger ^ elem(N - 1), A2 ^ elem(N - 1))}, !IO),
    array.shrink(2, Bigger, Two),
    io.write_line(Two, !IO),
    % Records in pairs in a list, sorted, which compares them.
    Pairs = list.map((func(P) = P ^ at - P ^ colour), array.to_list(A2)),
    list.sort_and_remove_dups(Pairs, Members),
    io.write_line(list.length(Members) : int, !IO),
    list.take_upto(3, Members, First),
    io.write_line(First, !IO),
    % Unification and comparison of whole records.
    compare(C, A2 ^ elem(1), A2 ^ elem(2)),
    io.write_line({C, same(A2 ^ elem(6), make(6)),
        same(A2 ^ elem(3), make(3))}, !IO),
    % An array filled from a list, and one of points alone.
    Ps = array.from_list([point(1i32, 2i32), point(-3i32, 4i32)]),
    Ps1 = array.append(Ps, array.init(2, point(0i32, 0i32))),
    io.write_line(Ps1, !IO),
    io.write_line(array.foldr((func(point(X, Y), S) =
        S * 10 + int32.to_int(X) - int32.to_int(Y)), Ps1, 0), !IO).

