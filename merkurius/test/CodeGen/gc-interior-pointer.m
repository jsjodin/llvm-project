% A pointer into an array's cell past its header, as the elements' address
% that LLVM computes once before a loop and carries round it, live across
% safepoints whose bundles the Merkurius code did not give it. The statepoint
% pass lists such a value in the bundle, and the rewrite takes whatever a
% bundle lists to be a base: it would pair the pointer with itself, and a
% collection would take it for the start of a cell. The pass lists the cell
% instead, and the pointer moves by its offset from it. Here srt's loop over
% the passes carries both arrays' elements across the polls and calls of
% merge_runs and copy_range, which LLVM inlines; records make each element
% a cell, so every load allocates and every allocation can collect.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT:array([p(1, 2), p(2, 3), p(3, 1)])
% OUTPUT-NEXT:array([p(0, 10), p(0, 20), p(1, 3), p(1, 13), p(2, 6), p(2, 16), p(3, 9), p(3, 19), p(4, 2), p(4, 12), p(5, 5), p(5, 15), p(6, 8), p(6, 18), p(7, 1), p(7, 11), p(8, 4), p(8, 14), p(9, 7), p(9, 17)])
% OUTPUT-NEXT:array([p(0, 10), p(0, 20), p(1, 3), p(1, 13), p(2, 6), p(2, 16), p(3, 9), p(3, 19), p(4, 2), p(4, 12), p(5, 5), p(5, 15), p(6, 8), p(6, 18), p(7, 1), p(7, 11), p(8, 4), p(8, 14), p(9, 7), p(9, 17)])

:- module gc_interior_pointer.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, int, list.

:- type p
    --->    p(int, int).

    % array.sort's merge sort, compiled here as a user module's code.
    %
:- func srt(array(T)) = array(T).
:- mode srt(array_di) = array_uo is det.
:- pragma no_inline(srt/1).

srt(Array0) = Array :-
    size(Array0, Size),
    ( if Size < 2 then
        Array = Array0
    else
        array.copy(Array0, Scratch),
        sort_passes(1, Size, Array0, Scratch, Array)
    ).

    % Merges the sorted runs of Width elements of Src into runs of twice as
    % many in Dst0, then passes back from Dst to Src, until a run holds the
    % whole array. A bottom-up merge sort: each pass is a loop, so the stack
    % does not grow with the array.
    %
:- pred sort_passes(int::in, int::in, array(T)::array_di,
    array(T)::array_di, array(T)::array_uo) is det.

sort_passes(Width, Size, Src, Dst0, Sorted) :-
    ( if Width >= Size then
        Sorted = Src
    else
        merge_pass(Width, Size, 0, Src, Dst0, Dst),
        sort_passes(Width * 2, Size, Dst, Src, Sorted)
    ).

:- pred merge_pass(int::in, int::in, int::in, array(T)::array_ui,
    array(T)::array_di, array(T)::array_uo) is det.

merge_pass(Width, Size, Lo, Src, !Dst) :-
    ( if Lo >= Size then
        true
    else
        Mid = int.min(Lo + Width, Size),
        Hi = int.min(Lo + 2 * Width, Size),
        merge_runs(Src, Lo, Mid, Mid, Hi, Lo, !Dst),
        merge_pass(Width, Size, Hi, Src, !Dst)
    ).

    % Merges Src's elements I to IEnd - 1 and J to JEnd - 1 into Dst from K,
    % taking the first run's element when two compare equal.
    %
:- pred merge_runs(array(T)::array_ui, int::in, int::in, int::in, int::in,
    int::in, array(T)::array_di, array(T)::array_uo) is det.

merge_runs(Src, I, IEnd, J, JEnd, K, !Dst) :-
    ( if I < IEnd, J < JEnd then
        X = Src ^ unsafe_elem(I),
        Y = Src ^ unsafe_elem(J),
        compare(R, Y, X),
        ( if R = (<) then
            unsafe_set(K, Y, !Dst),
            merge_runs(Src, I, IEnd, J + 1, JEnd, K + 1, !Dst)
        else
            unsafe_set(K, X, !Dst),
            merge_runs(Src, I + 1, IEnd, J, JEnd, K + 1, !Dst)
        )
    else if I < IEnd then
        copy_range(Src, I, IEnd, K, !Dst)
    else
        copy_range(Src, J, JEnd, K, !Dst)
    ).

    % Copies Src's elements I to End - 1 into Dst from K.
    %
:- pred copy_range(array(T)::array_ui, int::in, int::in, int::in,
    array(T)::array_di, array(T)::array_uo) is det.

copy_range(Src, I, End, K, !Dst) :-
    ( if I >= End then
        true
    else
        unsafe_set(K, Src ^ unsafe_elem(I), !Dst),
        copy_range(Src, I + 1, End, K + 1, !Dst)
    ).

main(!IO) :-
    A = array([p(3, 1), p(1, 2), p(2, 3)]),
    io.write_line(srt(A), !IO),
    list.map(make_p, 1 .. 20, Ps),
    io.write_line(srt(array(Ps)), !IO),
    io.write_line(array.sort(array(Ps)), !IO).

:- pred make_p(int::in, p::out) is det.

make_p(I, p((I * 7) mod 10, I)).
