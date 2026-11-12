% At -O2 the optimizer moves loads of a cell's fields past calls, so a
% pointer loaded from a field is live at a safepoint where merk's liveness
% listed only the cell, and t3's unification compares c3_1's fields through
% calls that can collect. The statepoint pass gives such a root the type of
% the field it was loaded from, which lowering records for each field it
% reads (kFieldTypes), and an argument the type lowering gives it
% (kRootTypeAttrName); without them the compile stopped with "a root ... has
% no type at a safepoint". The types are from the codegen oracle's
% generate-layout-programs.py, which found it.
%
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 | FileCheck %s --match-full-lines --strict-whitespace
%
% CHECK:same
% CHECK-NEXT:different
% CHECK-NEXT:c3_1(c2_0(c1_9(c1_0(2), ['a', 'b']), c1_5([2]), 'z'), [c3_1(c2_0(c1_9(c1_0(1), ['a', 'b']), c1_5([1]), 'z'), [c3_5(3u32), c3_0(c3_2('q'))], [1, 2]), c3_0(c3_2('q'))], [2, 3])

:- module sunk_field_roots.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, float, int, int16, int8, list, uint32, uint8.

:- type t1
    --->    c1_0(int)
    ;       c1_5(list(int))
    ;       c1_4(int16, {int, char}, t1)
    ;       c1_1(uint8, float)
    ;       c1_10({int, int8})
    ;       c1_8(int, float, float)
    ;       c1_3({int, uint32}, {int, float})
    ;       c1_9(t1, list(char))
    ;       c1_12(t1, list(char), {int, float})
    ;       k1_1
    ;       c1_7(int, {int, int})
    ;       c1_11(int)
    ;       c1_2({int, int}, char)
    ;       c1_6(list(int), int, int)
    ;       k1_2
    ;       k1_0.

:- type t2
    --->    k2_0
    ;       c2_0(t1, t1, char).

:- type t3
    --->    c3_5(uint32)
    ;       c3_0(t3)
    ;       c3_6(t3)
    ;       c3_8(int, char, list(int))
    ;       c3_2(char)
    ;       c3_3(float, uint32, int)
    ;       c3_9(list(int), char)
    ;       c3_1(t2, list(t3), list(int))
    ;       c3_4(float, int)
    ;       c3_7({int, t1}).


:- func make(int) = t3.
make(N) =
    ( if N =< 0 then
        c3_5(3u32)
    else
        c3_1(c2_0(c1_9(c1_0(N), ['a', 'b']), c1_5([N]), 'z'),
            [make(N - 1), c3_0(c3_2('q'))], [N, N + 1])
    ).

:- pred same(t3::in, t3::in) is semidet.
:- pragma no_inline(same/2).
same(A, B) :- A = B.

main(!IO) :-
    A = make(3),
    B = make(3),
    C = make(2),
    ( if same(A, B) then io.write_string("same\n", !IO)
    else io.write_string("different\n", !IO) ),
    ( if same(A, C) then io.write_string("same\n", !IO)
    else io.write_string("different\n", !IO) ),
    io.write_line(C, !IO).
