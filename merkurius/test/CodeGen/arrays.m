% The library's array module, which code generation builds in: an array
% is a pointer to its size and its elements, a word each; a set updates it
% in place, as uniqueness checking allows. The output is mmc's for the same
% program. The predicates that read an array take it array_ui, as merkc's
% sound declarations of the library want (see the design document's
% register, "Arrays' uniqueness").
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.O2 && %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% MERK-LABEL: merk.proc @arrays.main_2_0()
% MERK: merk.throw_if %{{.*}}, "Software Error: predicate `array.init'/3: Unexpected: negative size"
% MERK-NEXT: merk.array_new %{{.*}}, %{{.*}} : i64 -> <i64>
% MERK-LABEL: merk.proc private @arrays.squares_3_0(%arg0: i64, %arg1: !merk.array<i64>) -> !merk.array<i64>
% MERK: merk.check_index %arg1[%arg0], "array.set" : <i64>
% MERK-NEXT: merk.array_set %arg1[%arg0], %{{.*}} : <i64>, i64
% MERK-LABEL: merk.proc private @arrays.show_4_0(%arg0: !merk.array<i64>, %arg1: i64)
% MERK: %[[SIZE:.*]] = merk.array_size %arg0 : <i64>
% MERK-NEXT: %[[IN:.*]] = arith.cmpi ult, %arg1, %[[SIZE]] : i64
% MERK-NEXT: merk.guard %[[IN]]
% MERK-LABEL: merk.proc private @arrays.sum_4_0(%arg0: !merk.array<i64>, %arg1: i64, %arg2: i64) -> i64
% MERK: merk.check_index %arg0[%arg1], "array.lookup" : <i64>
% MERK-NEXT: merk.array_get %arg0[%arg1] : <i64> -> i64

% OUTPUT:0 1 4 9 16 
% OUTPUT-NEXT:30
% OUTPUT-NEXT:0 4 4 0
% OUTPUT-NEXT:out
% OUTPUT-NEXT:empty
% OUTPUT-NEXT:100 16 4 9 1 
% OUTPUT-NEXT:0 1 4 9 16 
% OUTPUT-NEXT:100 7 7 9 1 
% OUTPUT-NEXT:100 7 7 9 1 -1 -1 
% OUTPUT-NEXT:100 7 7 
% OUTPUT-NEXT:9 9 9 
% OUTPUT-NEXT:1 9 9 
% OUTPUT-NEXT:9 9 9 
% OUTPUT-NEXT:no set
% OUTPUT-NEXT:10
% OUTPUT-NEXT:3.75
% OUTPUT-NEXT:abc
% OUTPUT-NEXT:x
% OUTPUT-NOT:{{.}}

:- module arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, float, string.

:- pred sum(array(int)::array_ui, int::in, int::in, int::out) is det.
sum(A, I, S0, S) :-
    ( if I < array.size(A) then
        sum(A, I + 1, S0 + A ^ elem(I), S)
    else
        S = S0
    ).

:- pred squares(int::in, array(int)::array_di, array(int)::array_uo) is det.
squares(I, !A) :-
    ( if I < array.size(!.A) then
        array.set(I, I * I, !A),
        squares(I + 1, !A)
    else
        true
    ).

:- pred show(array(int)::array_ui, int::in, io::di, io::uo) is det.
show(A, I, !IO) :-
    ( if array.semidet_lookup(A, I, X) then
        io.write_int(X, !IO),
        io.write_string(" ", !IO),
        show(A, I + 1, !IO)
    else
        io.nl(!IO)
    ).

main(!IO) :-
    A0 = array.init(5, 0),
    squares(0, A0, A1),
    show(A1, 0, !IO),
    sum(A1, 0, 0, S),
    io.write_int(S, !IO), io.nl(!IO),
    array.bounds(A1, Lo, Hi),
    io.write_int(Lo, !IO), io.write_string(" ", !IO),
    io.write_int(Hi, !IO), io.write_string(" ", !IO),
    io.write_int(array.max(A1), !IO), io.write_string(" ", !IO),
    io.write_int(array.min(A1), !IO), io.nl(!IO),
    ( if array.in_bounds(A1, 5) then io.write_string("in\n", !IO) else io.write_string("out\n", !IO) ),
    E = array.make_empty_array : array(int),
    ( if array.is_empty(E) then io.write_string("empty\n", !IO) else io.write_string("not\n", !IO) ),
    array.copy(A1, B0),
    B1 = B0 ^ elem(0) := 100,
    array.swap(1, 4, B1, B2),
    show(B2, 0, !IO),
    show(A1, 0, !IO),
    array.fill_range(7, 1, 2, B2, B3),
    show(B3, 0, !IO),
    array.resize(7, -1, B3, B4),
    show(B4, 0, !IO),
    B5 = array.shrink(B4, 3),
    show(B5, 0, !IO),
    array.fill(9, B5, B6),
    show(B6, 0, !IO),
    C = array.slow_set(B6, 0, 1),
    show(C, 0, !IO),
    show(B6, 0, !IO),
    ( if array.semidet_set(3, 0, C, C1) then show(C1, 0, !IO) else io.write_string("no set\n", !IO) ),
    io.write_int(array.lookup(C, 2) + C ^ unsafe_elem(0), !IO), io.nl(!IO),
    F0 = array.init(3, 1.5),
    array.set(1, 2.25, F0, F1),
    io.write_float(F1 ^ elem(0) + F1 ^ elem(1), !IO), io.nl(!IO),
    S0 = array.init(2, "a"),
    S1 = array.set(S0, 1, "bc"),
    io.write_string(S1 ^ elem(0), !IO), io.write_string(S1 ^ elem(1), !IO), io.nl(!IO),
    Ch = array.init(2, 'x'),
    io.write_char(array.lookup(Ch, 1), !IO), io.nl(!IO).
