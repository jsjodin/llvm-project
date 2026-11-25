% MANUAL: User-defined equality and comparison
% MANUAL: Type classes
% Interaction: a type with an equality of its own, in instance methods that
% make and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 1 members, not equal to [1, 2, 3] | offset 2 members, not equal to [1, 2, 3]
% OUTPUT-NEXT:2: 2 members, not equal to [1, 2, 3] | offset 3 members, equal to [1, 2, 3]
% OUTPUT-NEXT:3: 3 members, equal to [1, 2, 3] | offset 4 members, not equal to [1, 2, 3]
% OUTPUT-NEXT:4: 4 members, not equal to [1, 2, 3] | offset 5 members, not equal to [1, 2, 3]
% OUTPUT-NOT:{{.}}
:- module typeclass_method_user_defined_equality.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

    % A set as a list in any order, equal by its members.
:- type fdata
    --->    set(list(int))
    where equality is same_members.

:- pred same_members(fdata::in, fdata::in) is semidet.

:- func make(int) = fdata.
:- func use(fdata) = string.

same_members(A, B) :-
    promise_equivalent_solutions [SA, SB] (
        A = set(LA),
        B = set(LB),
        list.sort_and_remove_dups(LA, SA),
        list.sort_and_remove_dups(LB, SB)
    ),
    SA = SB.

make(N) = set(list.reverse(1 .. N) ++ [1]).

use(D) = S :-
    % Built first, so that the unification compares two values, by the
    % type's equality.
    Expected = set([3, 2, 1]),
    ( if D = Expected then
        Equal = "equal to [1, 2, 3]"
    else
        Equal = "not equal to [1, 2, 3]"
    ),
    promise_equivalent_solutions [Count] (
        D = set(L),
        list.sort_and_remove_dups(L, Members),
        Count = list.length(Members)
    ),
    S = string.int_to_string(Count) ++ " members, " ++ Equal.

:- typeclass runner(T) where [
    func produce(T, int) = fdata,
    func consume(T, fdata) = string
].

:- type plain
    --->    plain.

:- type offset
    --->    offset(int).

:- instance runner(plain) where [
    produce(plain, N) = make(N),
    consume(plain, V) = use(V)
].

:- instance runner(offset) where [
    produce(offset(K), N) = make(N + K),
    consume(offset(_), V) = "offset " ++ use(V)
].

    % Through a constraint, so that the methods are called by the
    % dictionary.
:- func via(T, int) = string <= runner(T).
:- pragma no_inline(via/2).

via(R, N) = consume(R, produce(R, N)).

:- func run(int) = string.

run(N) = via(plain, N) ++ " | " ++ via(offset(1), N).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
