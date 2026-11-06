% An if-then-else whose condition has more than one solution: its then
% part runs for each of the condition's, and its else part only if the
% condition has none, even when the then part fails (a soft cut). It is
% merk.soft_ite in merk.each, and in merk.commit where only its first
% solution is wanted. The output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module soft_cuts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module int, list.

:- pred member2(int::out, list(int)::in) is nondet.
member2(X, [H | T]) :-
    ( X = H ; member2(X, T) ).

% Nondet: every big element doubled, or 0 if none.
:- pred big_or_zero(list(int)::in, int::out) is multi.
big_or_zero(L, Y) :-
    ( if member2(X, L), X > 4 then Y = X * 2 else Y = 0 ).
% MERK-LABEL: merk.proc private @soft_cuts.big_or_zero_2_0(
% MERK-NEXT: merk.each {
% MERK-NEXT: merk.soft_ite {
% MERK-NEXT: merk.nondet_call @soft_cuts.member2_2_0(
% MERK: merk.solution %{{.*}} : i64
% MERK: } then {
% MERK-NEXT: ^bb0(%[[X:.*]]: i64):
% MERK: %[[Y:.*]] = arith.muli %[[X]],
% MERK-NEXT: merk.solution %[[Y]] : i64
% MERK-NEXT: } else {
% MERK-NEXT: %[[ZERO:.*]] = merk.constant 0 : i64
% MERK-NEXT: merk.solution %[[ZERO]] : i64

% The then part can fail: no else once the condition succeeded.
:- pred odd_big(list(int)::in, int::out) is nondet.
odd_big(L, Y) :-
    ( if member2(X, L), X > 4 then X mod 2 = 1, Y = X else Y = -1 ).

% A goal after the if-then-else, which backtracks into it.
:- pred after(list(int)::in, int::out) is nondet.
after(L, Z) :-
    ( if member2(X, L) then Y = X + 1 else Y = 100 ),
    Y > 3,
    Z = Y * 10.

% Semidet with no outputs from the if-then-else: committed.
:- pred has_then(list(int)::in) is semidet.
has_then(L) :-
    ( if member2(X, L), X > 4 then X < 9 else true ).
% MERK-LABEL: merk.proc private @soft_cuts.has_then_1_0(
% MERK-NEXT: merk.commit {
% MERK-NEXT: merk.each {
% MERK-NEXT: merk.soft_ite {

% Nested: a soft cut in the condition of another.
:- pred nested(list(int)::in, list(int)::in, int::out) is multi.
nested(A, B, R) :-
    ( if ( if member2(X, A) then member2(Y, B), Y > X else Y = 0 ) then
        R = Y
    else
        R = -5
    ).


main(!IO) :-
    ( if big_or_zero([1, 5, 7], A), A > 12 then show(A, !IO) else none(!IO) ),
    ( if big_or_zero([1, 5, 7], B), B > 0 then show(B, !IO) else none(!IO) ),
    ( if big_or_zero([1, 2], C), C > -1 then show(C, !IO) else none(!IO) ),
    ( if odd_big([5, 6, 7], D), D > 6 then show(D, !IO) else none(!IO) ),
    ( if odd_big([6, 8], E), E > -10 then show(E, !IO) else none(!IO) ),
    ( if odd_big([1], F), F > -10 then show(F, !IO) else none(!IO) ),
    ( if after([1, 3, 5], G), G > 45 then show(G, !IO) else none(!IO) ),
    ( if after([], H), H > 0 then show(H, !IO) else none(!IO) ),
    ( if has_then([1, 6]) then io.write_string("yes\n", !IO) else io.write_string("no\n", !IO) ),
    ( if has_then([1, 10]) then io.write_string("yes\n", !IO) else io.write_string("no\n", !IO) ),
    ( if has_then([1]) then io.write_string("yes\n", !IO) else io.write_string("no\n", !IO) ),
    ( if nested([1, 4], [2, 5], I), I > 4 then show(I, !IO) else none(!IO) ),
    ( if nested([], [2, 5], J), J > -1 then show(J, !IO) else none(!IO) ),
    ( if nested([9], [2, 5], K), K > -9 then show(K, !IO) else none(!IO) ),
    big_or_zero([3, 8], L), show(L, !IO),
    ( if member2(M, [3, 8]), M > 5 then show(M, !IO) else none(!IO) ).

:- pred show(int::in, io::di, io::uo) is det.
show(X, !IO) :- io.write_int(X, !IO), io.nl(!IO).
:- pred none(io::di, io::uo) is det.
none(!IO) :- io.write_string("none\n", !IO).

% OUTPUT:14
% OUTPUT-NEXT:10
% OUTPUT-NEXT:0
% OUTPUT-NEXT:7
% OUTPUT-NEXT:none
% OUTPUT-NEXT:-1
% OUTPUT-NEXT:60
% OUTPUT-NEXT:1000
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:no
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:5
% OUTPUT-NEXT:0
% OUTPUT-NEXT:-5
% OUTPUT-NEXT:16
% OUTPUT-NEXT:8
