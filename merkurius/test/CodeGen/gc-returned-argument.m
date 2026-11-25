% An array that a call is passed and returns, which LLVM then uses in place
% of the call's result: it infers `returned` on copy_range's Dst, so app's
% second call is passed C0, live across the first, rather than C1. A
% collection in the first call moves the array, which copy_range holds as a
% typed root; app's slot must follow it, or the second call writes into the
% cell the collection freed. The records make each element a cell, so that
% unsafe_set allocates and every allocation can collect.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT:array([p(3, 1), p(1, 2), p(3, 1), p(1, 2)])
% OUTPUT-NEXT:array([p(5, 6), p(7, 8), p(1, 2), p(3, 4), p(5, 6), p(7, 8)])

:- module gc_returned_argument.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, int, list.

:- type p
    --->    p(int, int).

:- func app(array(T)::in, array(T)::in) = (array(T)::array_uo) is det.
:- pragma no_inline(app/2).

app(A, B) = C :-
    SizeA = size(A),
    SizeB = size(B),
    ( if SizeA > 0 then
        init(SizeA + SizeB, A ^ unsafe_elem(0), C0),
        copy_range(B, 0, SizeB, SizeA, C0, C1),
        copy_range(A, 1, SizeA, 1, C1, C)
    else if SizeB > 0 then
        init(SizeB, B ^ unsafe_elem(0), C0),
        copy_range(B, 1, SizeB, 1, C0, C)
    else
        make_empty_array(C)
    ).

:- pred copy_range(array(T)::in, int::in, int::in, int::in,
    array(T)::array_di, array(T)::array_uo) is det.

copy_range(Src, I, End, K, !Dst) :-
    ( if I >= End then
        true
    else
        unsafe_set(K, Src ^ unsafe_elem(I), !Dst),
        copy_range(Src, I + 1, End, K + 1, !Dst)
    ).

main(!IO) :-
    A = array([p(3, 1), p(1, 2)]),
    io.write_line(app(A, A), !IO),
    B = array([p(1, 2), p(3, 4), p(5, 6), p(7, 8)]),
    io.write_line(app(array([p(5, 6), p(7, 8)]), B), !IO).
