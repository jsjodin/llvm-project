% MANUAL: User-defined equality and comparison
% The manual's set as an unsorted list with a user-defined equality,
% defined by a subset predicate: (in, in) unifications of sets, including
% those of a type that contains them, are computed by it, and a
% deconstruction that cannot fail is cc_multi, so it is made in a
% promise_equivalent_solutions scope.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[1,2] [2,1] equal
% OUTPUT-NEXT:[1,2] [2,1,1] equal
% OUTPUT-NEXT:[1,2] [1,3] different
% OUTPUT-NEXT:named equal
% OUTPUT-NEXT:2
% OUTPUT-NOT:{{.}}

:- module run_set_equals.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
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

:- type named ---> named(string, set(int)).

:- pred test(string::in, set(int)::in, set(int)::in, io::di, io::uo) is det.
test(Name, A, B, !IO) :-
    io.write_string(Name, !IO),
    ( if A = B then
        io.write_string(" equal\n", !IO)
    else
        io.write_string(" different\n", !IO)
    ).

:- pred test_named(named::in, named::in, io::di, io::uo) is det.
test_named(A, B, !IO) :-
    ( if A = B then
        io.write_string("named equal\n", !IO)
    else
        io.write_string("named different\n", !IO)
    ).

main(!IO) :-
    test("[1,2] [2,1]", set([1, 2]), set([2, 1]), !IO),
    test("[1,2] [2,1,1]", set([1, 2]), set([2, 1, 1]), !IO),
    test("[1,2] [1,3]", set([1, 2]), set([1, 3]), !IO),
    test_named(named("a", set([3, 4])), named("a", set([4, 3])), !IO),
    S = set([5, 6]),
    promise_equivalent_solutions [L] (
        S = set(L)
    ),
    list.length(L, N),
    io.write_line(N, !IO).
