% A disjunction where only the first solution is wanted, whose branches
% have more than one: in a committed choice procedure, it is in
% merk.commit, as a disjunction with more than one solution, whose first
% is the first branch's first, or else the next branch's; with no outputs,
% each branch is such a goal of its own, in an if-then-else chain. The
% output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module committed_disjunctions.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module int, list.

:- pred member2(int::out, list(int)::in) is nondet.
member2(X, [H | T]) :-
    ( X = H ; member2(X, T) ).

:- pred first(list(int)::in, int::out) is cc_multi.
first(L, X) :- ( member2(X, L) ; X = 0 ).
% MERK-LABEL: merk.proc private @committed_disjunctions.first_2_0(
% MERK-NEXT: %[[FIRST:.*]] = merk.commit -> i64 {
% MERK-NEXT: merk.each {
% MERK-NEXT: merk.disj {
% MERK-NEXT: merk.nondet_call @committed_disjunctions.member2_2_0(
% MERK: merk.succeed %[[FIRST]] : i64

:- pred final(list(int)::in, int::out) is cc_multi.
final(L, X) :- ( L = [], X = 1 ; member2(X, L) ; X = 2 ).

:- pred first_big(list(int)::in) is semidet.
first_big(L) :- ( member2(X, L), X > 4 ; L = [] ).
% MERK-LABEL: merk.proc private @committed_disjunctions.first_big_1_0(
% MERK-NEXT: merk.ite {
% MERK-NEXT: merk.commit {
% MERK-NEXT: merk.nondet_call @committed_disjunctions.member2_2_0(

:- pred last_big(list(int)::in) is semidet.
last_big(L) :- ( L = [] ; member2(X, L), X > 4 ).

:- pred show(int::in, io::di, io::uo) is det.
show(X, !IO) :- io.write_int(X, !IO), io.nl(!IO).

:- pred test(string::in, bool_result::in, io::di, io::uo) is det.
:- type bool_result ---> yes ; no.
test(Name, R, !IO) :-
    io.write_string(Name, !IO),
    ( R = yes, io.write_string(": yes\n", !IO)
    ; R = no, io.write_string(": no\n", !IO)
    ).

main(!IO) :-
    first([7, 8], A), show(A, !IO),
    first([], B), show(B, !IO),
    final([], C), show(C, !IO),
    final([9], D), show(D, !IO),
    ( if first_big([1, 5]) then test("first_big", yes, !IO) else test("first_big", no, !IO) ),
    ( if first_big([1, 2]) then test("first_big", yes, !IO) else test("first_big", no, !IO) ),
    ( if first_big([]) then test("first_big", yes, !IO) else test("first_big", no, !IO) ),
    ( if last_big([1, 5]) then test("last_big", yes, !IO) else test("last_big", no, !IO) ),
    ( if last_big([1, 2]) then test("last_big", yes, !IO) else test("last_big", no, !IO) ).

% OUTPUT:7
% OUTPUT-NEXT:0
% OUTPUT-NEXT:1
% OUTPUT-NEXT:9
% OUTPUT-NEXT:first_big: yes
% OUTPUT-NEXT:first_big: no
% OUTPUT-NEXT:first_big: yes
% OUTPUT-NEXT:last_big: yes
% OUTPUT-NEXT:last_big: no
