% MANUAL: User-defined equality and comparison
% A type with only a comparison predicate: its unification is generated in
% terms of the comparison, as set_compare((=), S1, S2).
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:equal
% OUTPUT-NEXT:different
% OUTPUT-NEXT:'<'
% OUTPUT-NOT:{{.}}

:- module run_comparison_only.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- type set(T)
    --->    set(list(T))
            where comparison is set_compare.

:- pred set_compare(builtin.comparison_result::uo,
    set(T)::in, set(T)::in) is det.

set_compare(Result, Set1, Set2) :-
    promise_equivalent_solutions [Result] (
        set_compare_2(Set1, Set2, Result)
    ).

:- pred set_compare_2(set(T)::in, set(T)::in,
    builtin.comparison_result::uo) is cc_multi.

set_compare_2(set(List1), set(List2), Result) :-
    list.sort(List1, Sorted1),
    list.sort(List2, Sorted2),
    builtin.compare(Result, Sorted1, Sorted2).

:- pred test(set(int)::in, set(int)::in, io::di, io::uo) is det.
test(A, B, !IO) :-
    ( if A = B then
        io.write_string("equal\n", !IO)
    else
        io.write_string("different\n", !IO)
    ).

main(!IO) :-
    test(set([1, 2]), set([2, 1]), !IO),
    test(set([1, 2]), set([2, 3]), !IO),
    compare(R, set([1, 2]), set([2, 3])),
    io.write_line(R, !IO).
