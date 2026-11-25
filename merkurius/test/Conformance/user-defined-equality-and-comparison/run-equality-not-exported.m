% MANUAL: User-defined equality and comparison
% The manual says an exported type's equality predicate must be exported
% too, but mmc accepts one that is not ("Where mmc and the manual
% disagree"). Its unifications use it: here every two boxes are equal.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:equal
% OUTPUT-NOT:{{.}}

:- module run_equality_not_exported.
:- interface.
:- import_module io.

:- type box
    --->    box(int)
            where equality is box_equals.

:- pred main(io::di, io::uo) is det.

:- implementation.

:- pred box_equals(box::in, box::in) is semidet.
box_equals(_, _) :-
    semidet_true.

:- pred test(box::in, box::in, io::di, io::uo) is det.
test(A, B, !IO) :-
    ( if A = B then
        io.write_string("equal\n", !IO)
    else
        io.write_string("different\n", !IO)
    ).

main(!IO) :-
    test(box(1), box(2), !IO).
