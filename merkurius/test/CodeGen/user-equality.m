% User-defined equality and comparison (reference manual, "User-defined
% equality and comparison"): a type's unification procedure calls its
% equality predicate, or else its comparison's (=), and its comparison
% procedure calls its comparison predicate, or else throws software_error,
% whose text names the type, which the runtime writes from its type_info
% (merk.throw_if with a type_info). The procedures are the type's
% descriptor's entries, so a unification or comparison through a type
% variable calls them too. A nondet equality predicate's first solution is
% committed to. The output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -O2 %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: env MERK_GC_EVERY_ALLOC=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines

:- module user_equality.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list.

% Both, polymorphic.
:- type set(T)
    --->    set(list(T))
            where equality is set_equals, comparison is set_compare.
:- pred set_equals(set(T)::in, set(T)::in) is semidet.
set_equals(A, B) :-
    set_compare((=), A, B).
:- pred set_compare(comparison_result::uo, set(T)::in, set(T)::in) is det.
set_compare(R, A, B) :-
    promise_equivalent_solutions [R] (
        A = set(X),
        B = set(Y),
        list.sort_and_remove_dups(X, SX),
        list.sort_and_remove_dups(Y, SY),
        compare(R, SX, SY)
    ).

% Only a comparison: its unification is the comparison's (=).
:- type mod3
    --->    mod3(int)
            where comparison is mod3_compare.
:- pred mod3_compare(comparison_result::uo, mod3::in, mod3::in) is det.
mod3_compare(R, A, B) :-
    promise_equivalent_solutions [R] (
        A = mod3(X),
        B = mod3(Y),
        compare(R, X mod 3, Y mod 3)
    ).

% Only an equality, nondet, which the unification prunes; comparing
% throws.
:- type bag(T)
    --->    bag(list(T))
            where equality is bag_equals.
:- pred bag_equals(bag(T)::in, bag(T)::in) is nondet.
bag_equals(A, B) :-
    promise_equivalent_solutions [X, Y] (
        A = bag(X),
        B = bag(Y)
    ),
    list.member(E, X),
    list.member(E, Y).

% A type whose unification and comparison are its arguments'.
:- type pair_of(T) ---> pair_of(T, T).

:- pred eq(T::in, T::in, io::di, io::uo) is det.
eq(A, B, !IO) :-
    ( if A = B then io.write_string("equal\n", !IO)
    else io.write_string("different\n", !IO) ).

:- pred cmp(T::in, T::in, io::di, io::uo) is det.
cmp(A, B, !IO) :-
    compare(R, A, B),
    io.write_line(R, !IO).

:- pred direct(set(int)::in, set(int)::in, io::di, io::uo) is det.
direct(A, B, !IO) :-
    ( if A = B then io.write_string("equal\n", !IO)
    else io.write_string("different\n", !IO) ).

main(!IO) :-
    direct(set([1, 2, 2]), set([2, 1]), !IO),
    eq(set([1, 2]), set([2, 1]), !IO),
    eq(set([1, 2]), set([3]), !IO),
    cmp(set([3, 1]), set([1, 2]), !IO),
    eq(pair_of(set([1]), set([2, 1])), pair_of(set([1, 1]), set([1, 2])), !IO),
    cmp(pair_of(set([1]), set([2])), pair_of(set([1, 1]), set([1])), !IO),
    io.write_line(list.length(list.sort_and_remove_dups(
        [set([2, 1]), set([1, 2]), set([3])])) : int, !IO),
    eq(mod3(1), mod3(4), !IO),
    eq(mod3(1), mod3(5), !IO),
    cmp(mod3(2), mod3(4), !IO),
    eq(bag([1, 2]), bag([2, 3]), !IO),
    eq(bag(["a"]), bag(["b"]), !IO),
    ( if semidet_true then io.write_string("semidet_true\n", !IO) else true ),
    ( if semidet_fail then true else io.write_string("semidet_fail\n", !IO) ),
    try((pred(O::out) is det :- compare(O, bag(["x"]), bag(["y"]))), Result),
    (
        Result = succeeded(R),
        io.write_line(R, !IO)
    ;
        Result = exception(E),
        io.write_line(E, !IO)
    ).

% OUTPUT: equal
% OUTPUT-NEXT: equal
% OUTPUT-NEXT: different
% OUTPUT-NEXT: '>'
% OUTPUT-NEXT: equal
% OUTPUT-NEXT: '>'
% OUTPUT-NEXT: 2
% OUTPUT-NEXT: equal
% OUTPUT-NEXT: different
% OUTPUT-NEXT: '>'
% OUTPUT-NEXT: equal
% OUTPUT-NEXT: different
% OUTPUT-NEXT: semidet_true
% OUTPUT-NEXT: semidet_fail
% OUTPUT-NEXT: univ_cons(software_error("call to compare/3 for non-canonical type `user_equality.bag(string)\'"))
% OUTPUT-NOT: {{.}}

% MERK-LABEL: merk.proc private @"unify.user_equality.set(T0)"(
% MERK: merk.call @user_equality.set_equals_2_0(
% MERK-LABEL: merk.proc private @"unify.user_equality.bag(T0)"(
% MERK: merk.commit {
% MERK: merk.nondet_call @user_equality.bag_equals_2_0(
% MERK-LABEL: merk.proc private @"compare.user_equality.bag(T0)"(
% MERK: merk.throw_if %{{.*}}, "Software Error: call to compare/3 for non-canonical type `%T'" type %{{[0-9]+}}
