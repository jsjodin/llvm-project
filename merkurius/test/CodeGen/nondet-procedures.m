% Nondet and multi procedures (design document, Lowering, "Nondeterministic
% code"): a disjunction, a call of a nondet procedure, and a switch or an
% if-then-else whose branches have more than one solution, each give their
% solutions to the goals after them, in the procedures' order, and a
% committed choice procedure, or a goal with no outputs, takes the first.
% The output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module nondet_procedures.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module int, list.

:- type choice ---> a ; b ; c.

% The procedures print in the order of their names. A committed choice
% procedure's body, and a goal with no outputs, give their first solution
% from merk.commit.
% MERK-LABEL: merk.proc private @nondet_procedures.first_digit_1_0() -> i64
% MERK-NEXT: %[[FIRST:.*]] = merk.commit -> i64 {
% MERK-NEXT: merk.nondet_call @nondet_procedures.digit_1_0() {
% MERK: merk.solution %{{.*}} : i64
% MERK: merk.succeed %[[FIRST]] : i64
% MERK-LABEL: merk.proc private @nondet_procedures.has_big_1_0
% MERK-NEXT: merk.commit {
% MERK-NEXT: merk.nondet_call @nondet_procedures.member2_2_0(%arg0 : !merk.term<"list.list(int)">) {
% MERK: merk.guard
% MERK-NEXT: merk.solution
% MERK-NEXT: }
% MERK-NEXT: }
% MERK-NEXT: merk.succeed

% A disjunction in tail position, and a recursive call that passes its
% continuation on.
:- pred member2(int::out, list(int)::in) is nondet.
member2(X, [H | T]) :-
    ( X = H ; member2(X, T) ).
% MERK-LABEL: merk.proc private @nondet_procedures.member2_2_0(%arg0: !merk.term<"list.list(int)">) -> i64 attributes {determinism = #merk.determinism<nondet>} {
% MERK: %[[CELL:.*]]:2 = merk.deconstruct #merk.functor<"[|]", 2> %arg0
% MERK-NEXT: merk.each {
% MERK-NEXT: merk.disj {
% MERK-NEXT: merk.solution %[[CELL]]#0 : i64
% MERK-NEXT: }, {
% MERK-NEXT: merk.nondet_call @nondet_procedures.member2_2_0(%[[CELL]]#1 : !merk.term<"list.list(int)">) {
% MERK-NEXT: ^bb0(%[[X:.*]]: i64):
% MERK-NEXT: merk.solution %[[X]] : i64
% MERK-NEXT: }
% MERK-NEXT: }
% MERK-NEXT: } then {
% MERK-NEXT: ^bb0(%[[Y:.*]]: i64):
% MERK-NEXT: merk.succeed %[[Y]] : i64
% After outlining, the procedure takes a continuation and its environment,
% gives the head to the continuation, and if that does not stop, passes it
% on to the call of the second branch, a tail call.
% LLVM-LABEL: define internal tailcc i1 @nondet_procedures.member2_2_0(ptr addrspace(1) %0, ptr %1, ptr %2)
% LLVM: call tailcc i1 %1(ptr %2, i64 %{{.*}})
% LLVM: ret i1 true
% LLVM: musttail call tailcc i1 @nondet_procedures.member2_2_0(ptr addrspace(1) %{{.*}}, ptr %1, ptr %2)

% Polymorphic: the call passes a type_info.
:- pred elem(T::out, list(T)::in) is nondet.
elem(X, [H | T]) :-
    ( X = H ; elem(X, T) ).

% Facts, as a disjunction of unifications.
:- pred digit(int::out) is multi.
digit(1).
digit(2).
digit(3).

% Nondet calls in a conjunction, whose continuations use values from
% before them.
:- pred pair_sum(list(int)::in, int::in, int::out, int::out) is nondet.
pair_sum(L, S, X, Y) :-
    member2(X, L),
    member2(Y, L),
    X < Y,
    X + Y = S.
% MERK-LABEL: merk.proc private @nondet_procedures.pair_sum_4_0
% MERK-NEXT: merk.nondet_call @nondet_procedures.member2_2_0(%arg0 : !merk.term<"list.list(int)">) {
% MERK-NEXT: ^bb0(%[[X:.*]]: i64):
% MERK-NEXT: merk.nondet_call @nondet_procedures.member2_2_0(%arg0 : !merk.term<"list.list(int)">) {
% MERK-NEXT: ^bb0(%[[Y:.*]]: i64):
% MERK: merk.guard
% MERK: merk.guard
% MERK-NEXT: merk.succeed %[[X]], %[[Y]] : i64, i64

% A disjunction that is not last: the goals after it are its
% continuation.
:- pred scaled(int::out) is multi.
scaled(Y) :-
    ( X = 1 ; X = 2 ; X = 3 ),
    Y = X * 10.

% A switch whose cases have more than one solution.
:- pred pick(choice::in, int::out) is nondet.
pick(a, X) :- digit(X).
pick(b, X) :- member2(X, [7, 8]).
pick(c, _) :- fail.

% An if-then-else whose branches have more than one solution, and goals
% after it.
:- pred branchy(int::in, int::out) is multi.
branchy(N, Y) :-
    ( if N > 0 then
        digit(X)
    else
        ( X = 100 ; X = 200 )
    ),
    Y = X + N.

% A call in tail position of another nondet procedure.
:- pred big_member(int::out, list(int)::in) is nondet.
big_member(X, L) :-
    L = [_ | _],
    member2(X, L).

% No solutions.
:- pred none(int::out) is nondet.
none(X) :- member2(X, []).

% Committed choice procedures: the first solution.
:- pred first_pair(list(int)::in, int::in, int::out, int::out) is cc_nondet.
first_pair(L, S, X, Y) :- pair_sum(L, S, X, Y).

:- pred first_digit(int::out) is cc_multi.
first_digit(X) :- digit(Y), X = Y * 2.

:- pred first_scaled(int::in, int::out) is cc_nondet.
first_scaled(Min, Y) :- scaled(Y), Y > Min.

:- pred first_pick(choice::in, int::in, int::out) is cc_nondet.
first_pick(C, Min, X) :- pick(C, X), X > Min.

:- pred first_branchy(int::in, int::in, int::out) is cc_nondet.
first_branchy(N, Min, Y) :- branchy(N, Y), Y > Min.

:- pred first_elem(list(string)::in, string::out) is cc_nondet.
first_elem(L, X) :- elem(X, L), X \= "b".

:- pred first_big(list(int)::in, int::out) is cc_nondet.
first_big(L, X) :- big_member(X, L), X > 4.

:- pred first_none(int::out) is cc_nondet.
first_none(X) :- none(X).

% Goals with no outputs, of which only the first solution is wanted.
:- pred has_big(list(int)::in) is semidet.
has_big(L) :- member2(X, L), X > 4.

:- pred no_big(list(int)::in) is semidet.
no_big(L) :- \+ (member2(X, L), X > 4).

:- pred show(string::in, int::in, io::di, io::uo) is det.
show(Name, X, !IO) :-
    io.write_string(Name, !IO), io.write_string(": ", !IO),
    io.write_int(X, !IO), io.nl(!IO).

:- pred none(string::in, io::di, io::uo) is det.
none(Name, !IO) :-
    io.write_string(Name, !IO), io.write_string(": none\n", !IO).

:- pred bool(string::in, bool_result::in, io::di, io::uo) is det.
:- type bool_result ---> yes ; no.
bool(Name, R, !IO) :-
    io.write_string(Name, !IO),
    ( R = yes, io.write_string(": yes\n", !IO)
    ; R = no, io.write_string(": no\n", !IO)
    ).

main(!IO) :-
    L = [3, 1, 4, 1, 5, 9, 2, 6],
    ( if first_pair(L, 10, A, B) then
        show("pair", A, !IO), show("pair", B, !IO)
    else
        none("pair", !IO)
    ),
    ( if first_pair(L, 100, A1, B1) then
        show("pair", A1, !IO), show("pair", B1, !IO)
    else
        none("pair 100", !IO)
    ),
    first_digit(D),
    show("digit", D, !IO),
    ( if first_scaled(15, S) then show("scaled", S, !IO) else none("scaled", !IO) ),
    ( if first_pick(a, 2, P) then show("pick a", P, !IO) else none("pick a", !IO) ),
    ( if first_pick(b, 0, P1) then show("pick b", P1, !IO) else none("pick b", !IO) ),
    ( if first_pick(c, 0, P2) then show("pick c", P2, !IO) else none("pick c", !IO) ),
    ( if first_branchy(5, 6, Y) then show("branchy 5", Y, !IO) else none("branchy 5", !IO) ),
    ( if first_branchy(0, 150, Y1) then show("branchy 0", Y1, !IO) else none("branchy 0", !IO) ),
    ( if first_elem(["b", "c", "d"], E) then
        io.write_string("elem: ", !IO), io.write_string(E, !IO), io.nl(!IO)
    else
        none("elem", !IO)
    ),
    ( if first_big(L, G) then show("big", G, !IO) else none("big", !IO) ),
    ( if first_none(N) then show("none", N, !IO) else none("none", !IO) ),
    ( if has_big(L) then bool("has_big", yes, !IO) else bool("has_big", no, !IO) ),
    ( if has_big([1, 2]) then bool("has_big", yes, !IO) else bool("has_big", no, !IO) ),
    ( if no_big(L) then bool("no_big", yes, !IO) else bool("no_big", no, !IO) ),
    ( if no_big([1, 2]) then bool("no_big", yes, !IO) else bool("no_big", no, !IO) ).

% OUTPUT:pair: 1
% OUTPUT-NEXT:pair: 9
% OUTPUT-NEXT:pair 100: none
% OUTPUT-NEXT:digit: 2
% OUTPUT-NEXT:scaled: 20
% OUTPUT-NEXT:pick a: 3
% OUTPUT-NEXT:pick b: 7
% OUTPUT-NEXT:pick c: none
% OUTPUT-NEXT:branchy 5: 7
% OUTPUT-NEXT:branchy 0: 200
% OUTPUT-NEXT:elem: c
% OUTPUT-NEXT:big: 5
% OUTPUT-NEXT:none: none
% OUTPUT-NEXT:has_big: yes
% OUTPUT-NEXT:has_big: no
% OUTPUT-NEXT:no_big: no
% OUTPUT-NEXT:no_big: yes
