% MANUAL: User-defined equality and comparison
% The manual's set with both a user-defined equality and the comparison
% set_compare, which compares the sorted lists: compare/3 and sorting call
% it.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:'='
% OUTPUT-NEXT:'>'
% OUTPUT-NEXT:'<'
% OUTPUT-NEXT:equal
% OUTPUT-NEXT:4
% OUTPUT-NEXT:'='
% OUTPUT-NOT:{{.}}

:- module run_set_compare.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- type set(T)
    --->    set(list(T))
            where equality is set_equals, comparison is set_compare.

:- pred set_equals(set(T)::in, set(T)::in) is semidet.
set_equals(S1, S2) :-
    set_compare((=), S1, S2).

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
    compare(R1, set([2, 1]), set([1, 2])),
    io.write_line(R1, !IO),
    compare(R2, set([3, 1]), set([1, 2])),
    io.write_line(R2, !IO),
    compare(R3, set([1]), set([2, 1])),
    io.write_line(R3, !IO),
    test(set([1, 2]), set([2, 1]), !IO),
    Sorted = list.sort([set([3]), set([2, 1]), set([1, 2]), set([1])]),
    list.length(Sorted, N),
    io.write_line(N, !IO),
    compare(R4, [set([2, 1])], [set([1, 2])]),
    io.write_line(R4, !IO).
