% Arrays of narrow elements, int8 to uint32, characters and enumerations,
% hold each element as C does, in 1, 2 or 4 bytes, with the width in the
% top byte of the array's header (runtime/array.h). Code that knows the
% element type loads and stores them itself; polymorphic code, the
% library's map, foldl, sort and generate, and this module's own, reads the
% width from the header and makes arrays as wide as the type_info of what
% its type variable stands for says. Each runs on arrays made either way,
% is written by io.write, unified and compared, and kept by the collector,
% which copies a narrow array's words and follows none of them.
%
% The output is mmc's.
%
% RUN: merkc -fc1 -emit-llvm %s -o %t.ll
% RUN: FileCheck %s --check-prefix=LLVM < %t.ll
% RUN: FileCheck %s --check-prefix=NARROW < %t.ll
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 | FileCheck %s --match-full-lines --strict-whitespace

:- module narrow_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, bool, char, int, int8, int16, int32, list, string,
    uint8, uint16, uint32.

:- type colour
    --->    red
    ;       green
    ;       blue.

    % A term that holds narrow arrays, which the collector reaches through
    % its type_info.
:- type bag
    --->    bag(array(uint8), array(colour), string).

% A uint8 array is made 1 byte wide, a colour array 4, and a polymorphic
% one by its type_info.
% LLVM-LABEL: define {{.*}} @narrow_arrays.main_2_0(
% LLVM: call ptr addrspace(1) @merk_array_from_list({{.*}}, i64 1, i64 1)
% LLVM: call ptr addrspace(1) @merk_array_new(i64 {{.*}}, ptr addrspace(1) {{.*}}, i64 3)

main(!IO) :-
    I8 = array([-128i8, -1i8, 0i8, 1i8, 127i8]),
    U8 = array([0u8, 1u8, 128u8, 254u8, 255u8]),
    I16 = array([-32768i16, -2i16, 0i16, 32767i16]),
    U16 = array([0u16, 32768u16, 65535u16]),
    I32 = array([-2147483648i32, -3i32, 2147483647i32]),
    U32 = array([0u32, 2147483648u32, 4294967295u32]),
    Chars = array(['a', 'Z', 'ö', '€', '😀']),
    Colours = array([blue, red, green, blue]),
    Bools = array([yes, no, yes]),
    io.write_line(I8, !IO),
    io.write_line(U8, !IO),
    io.write_line(I16, !IO),
    io.write_line(U16, !IO),
    io.write_line(I32, !IO),
    io.write_line(U32, !IO),
    io.write_line(Chars, !IO),
    io.write_line(Colours, !IO),
    io.write_line(Bools, !IO),

    % Signed and unsigned arithmetic on elements, in code that knows the
    % types and in polymorphic folds.
    io.format("sums: %d %d %d %d %d %d\n",
        [i(sum_i8(I8)), i(sum_u8(U8)), i(int16.to_int(I16 ^ elem(1)) * 3),
        i(uint16.cast_to_int(U16 ^ elem(2)) + 1),
        i(int32.to_int(I32 ^ elem(0)) - 1),
        i(uint32.cast_to_int(U32 ^ elem(2)) + 1)], !IO),
    io.format("folds: %d %d %d\n",
        [i(array.foldl(add_i8, I8, 0)), i(array.foldl(add_u8, U8, 0)),
        i(array.foldl(count_blue, Colours, 0))], !IO),

    % Updates in place, a narrow store's bits only.
    U8b = ((array.copy(U8) ^ elem(0) := 200u8) ^ elem(4) := 7u8),
    I8b = array.copy(I8) ^ elem(2) := -100i8,
    I16b = array.copy(I16) ^ elem(3) := -7i16,
    Cb = array.copy(Colours) ^ elem(1) := green,
    io.write_line(bump_u8(U8b, 1), !IO),
    io.write_line(I8b, !IO),
    io.write_line(I16b, !IO),
    io.write_line(Cb, !IO),
    io.write_line(U8, !IO),

    % Polymorphic code: the library's map makes an array of what its
    % function returns, and generate makes the elements; this module's
    % insertion sort compares and moves them.
    io.write_line(array.map(negate_i8, I8), !IO),
    io.write_line(array.map(to_colour, U8), !IO),
    io.write_line(array.map(code, Chars), !IO),
    io.write_line(sorted(array([5u8, 3u8, 255u8, 0u8, 9u8])), !IO),
    io.write_line(sorted(array([3i16, -5i16, 0i16])), !IO),
    io.write_line(sorted(array.copy(Colours)), !IO),
    io.write_line(array.generate(6, square_u8), !IO),
    array.to_list(I32, L32),
    io.write_line(L32, !IO),

    % Copies, resizes, fills.
    io.write_line(array.resize(array.copy(U8), 8, 42u8), !IO),
    io.write_line(array.resize(array.copy(I16), 2, 0i16), !IO),
    io.write_line(array.shrink(array.copy(Chars), 3), !IO),
    array.init(5, 9u16, F0),
    array.fill(1u16, F0, F1),
    io.write_line(F1, !IO),
    io.write_line(array.from_reverse_list([1i8, 2i8, 3i8, -4i8]), !IO),

    % This module's polymorphic code, on arrays that monomorphic code made,
    % and the reverse.
    io.write_line(reversed(I8), !IO),
    io.write_line(reversed(Colours), !IO),
    io.write_line(reversed(U32), !IO),
    P = replicated(4, 250u8),
    io.write_line(P, !IO),
    io.format("replicated: %d %d\n",
        [i(uint8.to_int(P ^ elem(3))), i(sum_u8(P))], !IO),
    Q = replicated(3, '€'),
    io.write_line(Q, !IO),
    io.format("last: %c\n", [c(Q ^ elem(2))], !IO),

    % Unification and comparison by the elements' types: -1 is less than 0
    % as an int8, and 255 more than 0 as a uint8.
    ( if I8 = array.copy(I8) then
        io.write_string("equal\n", !IO)
    else
        io.write_string("unequal\n", !IO)
    ),
    ( if I8b = I8 then
        io.write_string("equal\n", !IO)
    else
        io.write_string("unequal\n", !IO)
    ),
    compare(R1, array([-1i8]), array([0i8])),
    compare(R2, array([255u8]), array([0u8])),
    compare(R3, array([-1i32, 5i32]), array([-1i32, 4i32])),
    compare(R4, array([red, blue]), array([green])),
    io.write_line({R1, R2, R3, R4}, !IO),

    % Arrays in a term, and many of them, kept across collections.
    Bags = list.map(make_bag, 1 .. 200),
    list.det_index1(Bags, 200, Last),
    io.write_line(Last, !IO),
    io.format("bags: %d\n", [i(list.foldl(weigh_bag, Bags, 0))], !IO).

:- func sum_i8(array(int8)) = int.
sum_i8(A) = sum_i8_from(A, 0, 0).

:- func sum_i8_from(array(int8), int, int) = int.
sum_i8_from(A, I, S) =
    ( if I < array.size(A) then
        sum_i8_from(A, I + 1, S + int8.to_int(A ^ elem(I)))
    else
        S
    ).

:- func sum_u8(array(uint8)) = int.
sum_u8(A) = sum_u8_from(A, 0, 0).

:- func sum_u8_from(array(uint8), int, int) = int.
sum_u8_from(A, I, S) =
    ( if I < array.size(A) then
        sum_u8_from(A, I + 1, S + uint8.to_int(A ^ elem(I)))
    else
        S
    ).

:- func add_i8(int8, int) = int.
add_i8(X, S) = S + int8.to_int(X).

:- func add_u8(uint8, int) = int.
add_u8(X, S) = S + uint8.to_int(X).

:- func count_blue(colour, int) = int.
count_blue(C, N) = ( if C = blue then N + 1 else N ).

:- func negate_i8(int8) = int8.
negate_i8(X) = -X.

:- func to_colour(uint8) = colour.
to_colour(X) = ( if X < 100u8 then red else if X < 200u8 then green else blue ).

:- func code(char) = int.
code(C) = char.to_int(C).

:- func square_u8(int) = uint8.
square_u8(I) = uint8.cast_from_int(I * I).

    % A copy with its elements in the reverse order.
    %
:- func reversed(array(T)) = array(T).
reversed(A) = B :-
    N = array.size(A),
    ( if N = 0 then
        B = A
    else
        array.init(N, A ^ elem(0), B0),
        B = reversed_from(A, N, 0, B0)
    ).

    % Element I of A is element N - 1 - I of B, from I on.
    %
:- func reversed_from(array(T), int, int, array(T)) = array(T).
:- mode reversed_from(in, in, in, array_di) = array_uo is det.
% A polymorphic element is read and written by the width in the header,
% merk.array_load and merk.array_store, inlined: a word, or 1, 2 or 4 bytes,
% zero-extended.
% LLVM-LABEL: define {{.*}} @narrow_arrays.reversed_from_f4_0(
% LLVM: %[[HEADER:.*]] = load i64, ptr addrspace(1) %1
% LLVM-NEXT: %[[SIZE:.*]] = and i64 %[[HEADER]], 72057594037927935
% LLVM: %[[HEADER2:.*]] = load i64, ptr addrspace(1) %1
% LLVM-NEXT: %[[CODE:.*]] = lshr i64 %[[HEADER2]], 56
% LLVM-NEXT: %[[WORDS:.*]] = icmp eq i64 %[[CODE]], 0
% LLVM: switch i64 %[[CODE]], label %{{.*}} [
% LLVM-NEXT: i64 1, label
% LLVM-NEXT: i64 2, label
% LLVM: load i8, ptr addrspace(1)
% LLVM-NEXT: zext i8 {{.*}} to i64
% LLVM: load i16, ptr addrspace(1)
% LLVM: load i32, ptr addrspace(1)
% LLVM: merk.array_load.exit:
% LLVM: store i8 {{.*}}, ptr addrspace(1)
% LLVM: store i16 {{.*}}, ptr addrspace(1)
% LLVM: store i32 {{.*}}, ptr addrspace(1)
reversed_from(A, N, I, B) =
    ( if I < N then
        reversed_from(A, N, I + 1, B ^ elem(N - 1 - I) := A ^ elem(I))
    else
        B
    ).

    % The narrow loads and stores of code that knows the element type.
    %
:- func bump_u8(array(uint8), int) = array(uint8).
:- mode bump_u8(array_di, in) = array_uo is det.
% NARROW-LABEL: define {{.*}} @narrow_arrays.bump_u8_f2_0(
% NARROW: %[[ELEMENTS:.*]] = getelementptr i64, ptr addrspace(1) %0, i32 1
% NARROW-NEXT: %[[AT:.*]] = getelementptr i8, ptr addrspace(1) %[[ELEMENTS]], i64 %1
% NARROW-NEXT: %[[X:.*]] = load i8, ptr addrspace(1) %[[AT]], align 1
% NARROW-NEXT: %[[Y:.*]] = add i8 %[[X]], 1
% NARROW: store i8 %[[Y]], ptr addrspace(1)
% NARROW-NEXT: ret
bump_u8(A, I) = A ^ elem(I) := A ^ elem(I) + 1u8.

    % The elements in ascending order, by an insertion sort.
    %
:- func sorted(array(T)) = array(T).
:- mode sorted(array_di) = array_uo is det.
sorted(A) = insert_from(1, A).

:- func insert_from(int, array(T)) = array(T).
:- mode insert_from(in, array_di) = array_uo is det.
insert_from(I, A) =
    ( if I < array.size(A) then
        insert_from(I + 1, sink(I, A))
    else
        A
    ).

    % Moves element I down past the greater elements before it.
    %
:- func sink(int, array(T)) = array(T).
:- mode sink(in, array_di) = array_uo is det.
sink(I, A0) = A :-
    ( if
        I > 0,
        compare((<), A0 ^ elem(I), A0 ^ elem(I - 1))
    then
        X = A0 ^ elem(I),
        Y = A0 ^ elem(I - 1),
        A1 = (A0 ^ elem(I) := Y) ^ elem(I - 1) := X,
        A = sink(I - 1, A1)
    else
        A = A0
    ).

:- func replicated(int, T) = array(T).
:- pragma no_inline(replicated/2).
replicated(N, X) = array.init(N, X).

:- func make_bag(int) = bag.
make_bag(I) = bag(
    array.init(I mod 13 + 1, uint8.cast_from_int(I mod 256)),
    array.init(I mod 3 + 1, ( if I mod 2 = 0 then green else blue )),
    string.int_to_string(I)).

:- func weigh_bag(bag, int) = int.
weigh_bag(bag(U, C, S), N) =
    N + sum_u8(U) + array.foldl(count_blue, C, 0) + string.length(S).

% CHECK:array([-128i8, -1i8, 0i8, 1i8, 127i8])
% CHECK-NEXT:array([0u8, 1u8, 128u8, 254u8, 255u8])
% CHECK-NEXT:array([-32768i16, -2i16, 0i16, 32767i16])
% CHECK-NEXT:array([0u16, 32768u16, 65535u16])
% CHECK-NEXT:array([-2147483648i32, -3i32, 2147483647i32])
% CHECK-NEXT:array([0u32, 2147483648u32, 4294967295u32])
% CHECK-NEXT:array(['a', 'Z', 'ö', '€', '😀'])
% CHECK-NEXT:array([blue, red, green, blue])
% CHECK-NEXT:array([yes, no, yes])
% CHECK-NEXT:sums: -1 638 -6 65536 -2147483649 4294967296
% CHECK-NEXT:folds: -1 638 2
% CHECK-NEXT:array([200u8, 2u8, 128u8, 254u8, 7u8])
% CHECK-NEXT:array([-128i8, -1i8, -100i8, 1i8, 127i8])
% CHECK-NEXT:array([-32768i16, -2i16, 0i16, -7i16])
% CHECK-NEXT:array([blue, green, green, blue])
% CHECK-NEXT:array([0u8, 1u8, 128u8, 254u8, 255u8])
% CHECK-NEXT:array([-128i8, 1i8, 0i8, -1i8, -127i8])
% CHECK-NEXT:array([red, red, green, blue, blue])
% CHECK-NEXT:array([97, 90, 246, 8364, 128512])
% CHECK-NEXT:array([0u8, 3u8, 5u8, 9u8, 255u8])
% CHECK-NEXT:array([-5i16, 0i16, 3i16])
% CHECK-NEXT:array([red, green, blue, blue])
% CHECK-NEXT:array([0u8, 1u8, 4u8, 9u8, 16u8, 25u8])
% CHECK-NEXT:[-2147483648i32, -3i32, 2147483647i32]
% CHECK-NEXT:array([0u8, 1u8, 128u8, 254u8, 255u8, 42u8, 42u8, 42u8])
% CHECK-NEXT:array([-32768i16, -2i16])
% CHECK-NEXT:array(['a', 'Z', 'ö'])
% CHECK-NEXT:array([1u16, 1u16, 1u16, 1u16, 1u16])
% CHECK-NEXT:array([-4i8, 3i8, 2i8, 1i8])
% CHECK-NEXT:array([127i8, 1i8, 0i8, -1i8, -128i8])
% CHECK-NEXT:array([blue, green, red, blue])
% CHECK-NEXT:array([4294967295u32, 2147483648u32, 0u32])
% CHECK-NEXT:array([250u8, 250u8, 250u8, 250u8])
% CHECK-NEXT:replicated: 250 1000
% CHECK-NEXT:array(['€', '€', '€'])
% CHECK-NEXT:last: €
% CHECK-NEXT:equal
% CHECK-NEXT:unequal
% CHECK-NEXT:{(<), (>), (>), (>)}
% CHECK-NEXT:bag(array([200u8, 200u8, 200u8, 200u8, 200u8, 200u8]), array([green, green, green]), "200")
% CHECK-NEXT:bags: 139992
