% MANUAL: User-defined equality and comparison
% A type with only an equality predicate: its generated comparison throws
% an exception of type exception.software_error when called, which names
% the type.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:equal
% OUTPUT-NEXT:univ_cons(software_error("call to compare/3 for non-canonical type `run_equality_only_compare_throws.set(int)\'"))
% OUTPUT-NOT:{{.}}

:- module run_equality_only_compare_throws.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module list.

:- type set(T)
    --->    set(list(T))
            where equality is set_equals.

:- pred set_equals(set(T)::in, set(T)::in) is semidet.
set_equals(S1, S2) :-
    subset(S1, S2),
    subset(S2, S1).

:- pred subset(set(T)::in, set(T)::in) is semidet.
subset(S1, S2) :-
    promise_equivalent_solutions [L1, L2] (
        S1 = set(L1),
        S2 = set(L2)
    ),
    all_in(L1, L2).

:- pred all_in(list(T)::in, list(T)::in) is semidet.
all_in([], _).
all_in([X | Xs], L) :-
    list.member(X, L),
    all_in(Xs, L).

:- import_module exception.

:- pred cmp(set(int)::in, set(int)::in, comparison_result::out) is det.
cmp(A, B, R) :-
    compare(R, A, B).

:- pred test(set(int)::in, set(int)::in, io::di, io::uo) is det.
test(A, B, !IO) :-
    ( if A = B then
        io.write_string("equal\n", !IO)
    else
        io.write_string("different\n", !IO)
    ).

main(!IO) :-
    test(set([1, 2]), set([2, 1]), !IO),
    try((pred(R::out) is det :- cmp(set([1]), set([2]), R)), Result),
    (
        Result = succeeded(Order),
        io.write_line(Order, !IO)
    ;
        Result = exception(E),
        io.write_line(E, !IO)
    ).
