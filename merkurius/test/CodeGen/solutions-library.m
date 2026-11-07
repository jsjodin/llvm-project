% The library's solutions module, compiled into libmerkstd: each predicate
% and function in each mode, with generators that have solutions and that
% have none, steps that are det and cc_multi, accumulators that are shared
% and unique, a do_while that stops early, a generator that uses a value
% from outside it, solutions collected inside a generator, and 100000
% solutions. Each call calls its mode's procedure. The output is mmc's for
% the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s 2>/dev/null | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 2>/dev/null && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 2>/dev/null && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module solutions_library.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module bool, float, int, list, pair, solutions, string.

:- pred show(T::in, io::di, io::uo) is det.
show(X, !IO) :- io.write(X, !IO), io.nl(!IO).

:- pred show_int(int::in, io::di, io::uo) is det.
show_int(N, !IO) :- show(N, !IO).

:- pred show_list(list(int)::in, io::di, io::uo) is det.
show_list(L, !IO) :- show(L, !IO).

:- pred g(int::out) is multi.
g(3). g(1). g(2). g(1).

:- pred none(int::out) is nondet.
none(X) :- g(X), X > 5.

:- pred digit(int::out) is multi.
digit(0). digit(1). digit(2). digit(3). digit(4).
digit(5). digit(6). digit(7). digit(8). digit(9).

:- pred names(string::out) is multi.
names("carol"). names("alice"). names("bob").

:- pred halves(float::out) is multi.
halves(0.5). halves(1.5). halves(0.25).

:- pred pairs(pair(int, string)::out) is multi.
pairs(2 - "b"). pairs(1 - "a"). pairs(2 - "a").

% Counts from 1 to N.
:- pred upto(int::in, int::out) is nondet.
upto(N, X) :- N >= 1, ( X = N ; upto(N - 1, X) ).

:- pred add(int::in, int::in, int::out) is det.
add(X, S, S + X).

:- pred add_cc(int::in, int::in, int::out) is cc_multi.
add_cc(X, S, R) :- ( R = S + X ; R = X + S ).

:- pred print_each(int::in, io::di, io::uo) is det.
print_each(X, !IO) :- io.write_int(X, !IO), io.write_char(',', !IO).

:- pred show_cc(int::in, io::di, io::uo) is cc_multi.
show_cc(X, !IO) :- io.write_int(X, !IO), io.write_char(',', !IO).

:- pred below(int::in, int::in, bool::out, int::in, int::out) is det.
below(Limit, X, More, S, S + X) :-
    ( if S + X >= Limit then More = no else More = yes ).

:- pred print_until(int::in, bool::out, io::di, io::uo) is det.
print_until(X, More, !IO) :-
    io.write_int(X, !IO),
    ( if X = 2 then More = no else More = yes ).

:- pred print_until_cc(int::in, bool::out, io::di, io::uo) is cc_multi.
print_until_cc(X, More, !IO) :-
    io.write_int(X, !IO),
    More = yes.

:- pred count_and_list(int::in, int::in, int::out, list(int)::in,
    list(int)::out) is det.
count_and_list(X, C, C + 1, L, [X | L]).

:- pred count_and_print(int::in, int::in, int::out, io::di, io::uo) is det.
count_and_print(X, C, C + 1, !IO) :- io.write_int(X, !IO).

:- pred count_and_show_cc(int::in, int::in, int::out, io::di, io::uo)
    is cc_multi.
count_and_show_cc(X, C, C + 1, !IO) :- io.write_int(X, !IO).

:- pred count_and_list_cc(int::in, int::in, int::out, list(int)::in,
    list(int)::out) is cc_multi.
count_and_list_cc(X, C, C1, L, [X | L]) :- ( C1 = C + 1 ; C1 = 1 + C ).

% The solutions of a goal that collects solutions itself.
:- pred sums(int::out) is nondet.
sums(S) :-
    digit(N), N < 3,
    solutions(upto(N), L),
    S = list.foldl(int.plus, L, 0).

% Mostly unique solutions, which the step takes before the generator
% backtracks over them.
:- pred fresh(list(int)::muo) is multi.
fresh([1, 2]). fresh([3]).

:- pred show_fresh(list(int)::mdi, io::di, io::uo) is det.
show_fresh(L, !IO) :- io.write(L, !IO), io.nl(!IO).

% A non-empty list from a multi generator.
:- pred first(list(int)::in(non_empty_list), int::out) is det.
first([X | _], X).

% MERK-LABEL: merk.proc @solutions_library.main_2_0(
% MERK: merk.call @solutions.solutions_2_0(
% MERK: merk.call @solutions.solutions_2_1(
% MERK: merk.call @solutions.solutions_f1_0(
% MERK: merk.call @solutions.solutions_f1_0(
% MERK: merk.call @solutions.solutions_f1_1(
% MERK: merk.call @solutions.solutions_2_0(
% MERK: merk.call @solutions.solutions_2_0(
% MERK: merk.call @solutions.solutions_2_0(
% MERK: merk.call @solutions.solutions_2_1(
% MERK: merk.call @solutions.solutions_2_1(
% MERK: merk.call @solutions.solutions_2_1(
% MERK: merk.call @solutions.unsorted_solutions_2_0(
% MERK: merk.call @solutions.unsorted_solutions_2_1(
% MERK: merk.call @solutions.unsorted_solutions_2_0(
% MERK: merk.call @solutions.aggregate_f3_0(
% MERK: merk.call @solutions.aggregate_f3_1(
% MERK: merk.call @solutions.aggregate_4_0(
% MERK: merk.call @solutions.aggregate_4_3(
% MERK: merk.call @solutions.aggregate_4_1(
% MERK: merk.call @solutions.aggregate_4_2(
% MERK: merk.call @solutions.aggregate2_6_0(
% MERK: merk.call @solutions.aggregate2_6_3(
% MERK: merk.call @solutions.aggregate2_6_1(
% MERK: merk.call @solutions.aggregate2_6_2(
% MERK: merk.call @solutions.unsorted_aggregate_4_0(
% MERK: merk.call @solutions.unsorted_aggregate_4_1(
% MERK: merk.call @solutions.unsorted_aggregate_4_2(
% MERK: merk.call @solutions.unsorted_aggregate_4_3(
% MERK: merk.call @solutions.unsorted_aggregate_4_5(
% MERK: merk.call @solutions.unsorted_aggregate_4_6(
% MERK: merk.call @solutions.unsorted_aggregate_4_7(
% MERK: merk.call @solutions.unsorted_aggregate_4_8(
% MERK: merk.call @solutions.unsorted_aggregate_4_7(
% MERK: merk.call @solutions.unsorted_aggregate_4_4(
% MERK: merk.call @solutions.unsorted_aggregate2_6_0(
% MERK: merk.call @solutions.unsorted_aggregate2_6_1(
% MERK: merk.call @solutions.unsorted_aggregate2_6_2(
% MERK: merk.call @solutions.unsorted_aggregate2_6_3(
% MERK: merk.call @solutions.unsorted_aggregate2_6_4(
% MERK: merk.call @solutions.unsorted_aggregate2_6_5(
% MERK: merk.call @solutions.unsorted_aggregate2_6_6(
% MERK: merk.call @solutions.unsorted_aggregate2_6_7(
% MERK: merk.call @solutions.do_while_4_0(
% MERK: merk.call @solutions.do_while_4_3(
% MERK: merk.call @solutions.do_while_4_1(
% MERK: merk.call @solutions.do_while_4_2(
% MERK: merk.call @solutions.do_while_4_4(
% MERK: merk.call @solutions.do_while_4_5(
% MERK: merk.call @solutions.do_while_4_3(
% OUTPUT:[1, 2, 3]
% OUTPUT-NEXT:1
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:[1, 2, 3]
% OUTPUT-NEXT:1
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:["alice", "bob", "carol"]
% OUTPUT-NEXT:[0.25, 0.5, 1.5]
% OUTPUT-NEXT:[1 - "a", 2 - "a", 2 - "b"]
% OUTPUT-NEXT:[0, 1, 3]
% OUTPUT-NEXT:[0, 1, 2, 3]
% OUTPUT-NEXT:100000
% OUTPUT-NEXT:[1, 2, 1, 3]
% OUTPUT-NEXT:1
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:["bob", "alice", "carol"]
% OUTPUT-NEXT:123
% OUTPUT-NEXT:7
% OUTPUT-NEXT:6
% OUTPUT-NEXT:5
% OUTPUT-NEXT:1,2,3,
% OUTPUT-EMPTY:
% OUTPUT-NEXT:{3, [3, 2, 1]}
% OUTPUT-NEXT:{0, []}
% OUTPUT-NEXT:123
% OUTPUT-NEXT:3
% OUTPUT-NEXT:0
% OUTPUT-NEXT:7
% OUTPUT-NEXT:7
% OUTPUT-NEXT:3,1,2,1,
% OUTPUT-NEXT:3,1,2,1,
% OUTPUT-EMPTY:
% OUTPUT-EMPTY:
% OUTPUT-NEXT:9
% OUTPUT-NEXT:9
% OUTPUT-NEXT:5000050000
% OUTPUT-NEXT:[1, 2]
% OUTPUT-NEXT:[3]
% OUTPUT-NEXT:{4, [1, 2, 1, 3]}
% OUTPUT-NEXT:{4, [1, 2, 1, 3]}
% OUTPUT-NEXT:3121
% OUTPUT-NEXT:4
% OUTPUT-NEXT:3121
% OUTPUT-NEXT:4
% OUTPUT-NEXT:{0, []}
% OUTPUT-NEXT:{0, []}
% OUTPUT-NEXT:0
% OUTPUT-NEXT:0
% OUTPUT-NEXT:10
% OUTPUT-NEXT:0
% OUTPUT-NEXT:312
% OUTPUT-NEXT:3121
% OUTPUT-EMPTY:
% OUTPUT-EMPTY:
% OUTPUT-NEXT:1099945
main(!IO) :-
    % solutions/2 and solutions/1, multi and nondet.
    solutions(g, S1), show(S1, !IO),
    first(S1, F1), show(F1, !IO),
    solutions(none, S2), show(S2, !IO),
    show_list(solutions(g), !IO),
    first(solutions(g), F2), show(F2, !IO),
    show_list(solutions(none), !IO),
    solutions(names, S3), show(S3, !IO),
    solutions(halves, S4), show(S4, !IO),
    solutions(pairs, S5), show(S5, !IO),
    solutions(sums, S6), show(S6, !IO),
    % A generator that uses a value from outside it.
    Limit = 4,
    solutions((pred(X::out) is nondet :- digit(X), X < Limit), S7), show(S7, !IO),
    solutions(upto(100000), Big), show_int(list.length(Big), !IO),
    % unsorted_solutions.
    unsorted_solutions(g, U1), show(U1, !IO),
    first(U1, F3), show(F3, !IO),
    unsorted_solutions(none, U2), show(U2, !IO),
    unsorted_solutions(names, U3), show(U3, !IO),
    % aggregate/3, aggregate/4, aggregate2/6.
    show_int(aggregate(g, (func(X, A) = X + 10 * A), 0), !IO),
    show_int(aggregate(none, (func(X, A) = X + 10 * A), 7), !IO),
    aggregate(g, add, 0, A1), show(A1, !IO),
    aggregate(none, add, 5, A2), show(A2, !IO),
    aggregate(g, print_each, !IO), io.nl(!IO),
    aggregate(none, print_each, !IO), io.nl(!IO),
    aggregate2(g, count_and_list, 0, C1, [], L1), show({C1, L1}, !IO),
    aggregate2(none, count_and_list, 0, C2, [], L2), show({C2, L2}, !IO),
    aggregate2(g, count_and_print, 0, C3, !IO), io.nl(!IO), show(C3, !IO),
    aggregate2(none, count_and_print, 0, C4, !IO), show(C4, !IO),
    % unsorted_aggregate/4, every kind of step.
    unsorted_aggregate(g, add, 0, A3), show(A3, !IO),
    unsorted_aggregate(g, add_cc, 0, A4), show(A4, !IO),
    unsorted_aggregate(g, print_each, !IO), io.nl(!IO),
    unsorted_aggregate(g, show_cc, !IO), io.nl(!IO),
    unsorted_aggregate(none, print_each, !IO), io.nl(!IO),
    unsorted_aggregate(none, show_cc, !IO), io.nl(!IO),
    unsorted_aggregate(none, add, 9, A5), show(A5, !IO),
    unsorted_aggregate(none, add_cc, 9, A6), show(A6, !IO),
    unsorted_aggregate(upto(100000), add, 0, A7), show(A7, !IO),
    unsorted_aggregate(fresh, show_fresh, !IO),
    % unsorted_aggregate2/6.
    unsorted_aggregate2(g, count_and_list, 0, C5, [], L5), show({C5, L5}, !IO),
    unsorted_aggregate2(g, count_and_list_cc, 0, C6, [], L6), show({C6, L6}, !IO),
    unsorted_aggregate2(g, count_and_print, 0, C7, !IO), io.nl(!IO), show(C7, !IO),
    unsorted_aggregate2(g, count_and_show_cc, 0, C8, !IO), io.nl(!IO), show(C8, !IO),
    unsorted_aggregate2(none, count_and_list, 0, C9, [], L9), show({C9, L9}, !IO),
    unsorted_aggregate2(none, count_and_list_cc, 0, C10, [], L10), show({C10, L10}, !IO),
    unsorted_aggregate2(none, count_and_print, 0, C11, !IO), show(C11, !IO),
    unsorted_aggregate2(none, count_and_show_cc, 0, C12, !IO), show(C12, !IO),
    % do_while/4: it stops when the step says no.
    do_while(digit, below(10), 0, D1), show(D1, !IO),
    do_while(none, below(10), 0, D2), show(D2, !IO),
    do_while(g, print_until, !IO), io.nl(!IO),
    do_while(g, print_until_cc, !IO), io.nl(!IO),
    do_while(none, print_until, !IO), io.nl(!IO),
    do_while(none, print_until_cc, !IO), io.nl(!IO),
    do_while(upto(100000), below(1000000), 0, D3), show(D3, !IO).
