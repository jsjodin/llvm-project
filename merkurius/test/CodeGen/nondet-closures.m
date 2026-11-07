% Closures of nondet and multi procedures and lambda expressions, and
% higher-order calls of them: the closure's entry calls the procedure's
% word copy with merk.nondet_call and gives its solutions as words, and a
% higher-order call is merk.nondet_call_closure, which outlining makes a
% semidet call of the entry with a continuation. The entry passes its
% continuation on, as a tail call. The output is mmc's for the same
% program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module nondet_closures.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module int, list.

:- pred member2(T::out, list(T)::in) is nondet.
member2(X, [H | T]) :-
    ( X = H ; member2(X, T) ).

:- pred elem(list(T)::in, T::out) is nondet.
elem(L, X) :- member2(X, L).

:- pred between(int::in, int::in, int::out) is nondet.
between(L, H, X) :-
    L =< H,
    ( X = L ; between(L + 1, H, X) ).

% Calls the generator, for each solution.
:- pred gen_filter(pred(int)::in(pred(out) is nondet), int::in, int::out)
    is nondet.
gen_filter(P, Min, X) :-
    P(X),
    X > Min.

% The first solution of a generator above Min, in a committed context.
:- pred first_above(pred(int)::in(pred(out) is nondet), int::in, int::out)
    is cc_nondet.
first_above(P, Min, X) :-
    gen_filter(P, Min, X).

% A multi closure.
:- pred pick(pred(int)::in(pred(out) is multi), int::out) is multi.
pick(P, X) :- P(X).

:- pred small(int::out) is multi.
small(1).
small(2).
small(3).

% A higher-order call whose closure takes an input.
:- pred apply_in(pred(int, int)::in(pred(in, out) is nondet), int::in,
    int::out) is nondet.
apply_in(P, A, B) :- P(A, B).

:- pred show(int::in, io::di, io::uo) is det.
show(X, !IO) :- io.write_int(X, !IO), io.nl(!IO).
:- pred none(io::di, io::uo) is det.
none(!IO) :- io.write_string("none\n", !IO).

main(!IO) :-
    L = [3, 8, 1, 9],
    % A closure of a polymorphic procedure, with a curried argument.
    P1 = elem(L),
    ( if first_above(P1, 5, A) then show(A, !IO) else none(!IO) ),
    ( if first_above(P1, 9, B) then show(B, !IO) else none(!IO) ),
    % A lambda expression that captures values.
    Lo = 10,
    P2 = (pred(X::out) is nondet :- between(Lo, Lo + 5, X), X mod 2 = 1),
    ( if first_above(P2, 11, C) then show(C, !IO) else none(!IO) ),
    % A lambda expression calling another closure.
    P3 = (pred(X::out) is nondet :- P1(Y), X = Y * 100),
    ( if first_above(P3, 500, D) then show(D, !IO) else none(!IO) ),
    % A multi closure, committed in main.
    pick(small, E), show(E, !IO),
    ( if pick(small, F), F > 1 then show(F, !IO) else none(!IO) ),
    % A closure with an input.
    P4 = (pred(I::in, O::out) is nondet :- member2(O, L), O > I),
    ( if apply_in(P4, 8, G) then show(G, !IO) else none(!IO) ),
    ( if apply_in(between(2), 4, H), H > 3 then show(H, !IO) else none(!IO) ),
    % A higher-order call directly in a soft cut's condition.
    ( if P1(K), K > 2 then show(K, !IO) else none(!IO) ).

% MERK-LABEL: merk.proc private @nondet_closures.elem_2_0$words$closure2(%arg0: !merk.closure) -> !merk.word attributes {determinism = #merk.determinism<nondet>} {
% MERK: merk.nondet_call @nondet_closures.elem_2_0$words(
% MERK-NEXT: ^bb0(%[[X:.*]]: !merk.word):
% MERK-NEXT: merk.succeed %[[X]] : !merk.word
% MERK-LABEL: merk.proc private @nondet_closures.main_2_0$lambda1(%arg0: !merk.closure) -> !merk.word attributes {determinism = #merk.determinism<nondet>} {
% MERK-NEXT: merk.nondet_call_closure %arg0() {
% MERK-NEXT: ^bb0(%[[Y:.*]]: !merk.word):

% LLVM-LABEL: define internal tailcc i1 @"nondet_closures.elem_2_0$words$closure2"(ptr %0, ptr %1, ptr %2)
% LLVM: musttail call tailcc i1 @"nondet_closures.elem_2_0$words"(ptr %{{.*}}, ptr %{{.*}}, ptr %1, ptr %2)
% LLVM-LABEL: define internal tailcc i1 @"nondet_closures.main_2_0$lambda0$closure1"(ptr %0, ptr %1, ptr %2)
% LLVM: musttail call tailcc i1 @"nondet_closures.main_2_0$lambda0"(i64 %{{.*}}, ptr %1, ptr %2)
% LLVM-LABEL: define internal tailcc i1 @"nondet_closures.main_2_0$lambda1"(ptr %0, ptr %1, ptr %2)
% LLVM: %[[CODE:.*]] = load ptr, ptr %0
% LLVM: call tailcc i1 %[[CODE]](ptr %0, ptr @"nondet_closures.main_2_0$lambda1.cont.0", ptr %{{.*}})

% OUTPUT:8
% OUTPUT-NEXT:none
% OUTPUT-NEXT:13
% OUTPUT-NEXT:800
% OUTPUT-NEXT:1
% OUTPUT-NEXT:2
% OUTPUT-NEXT:9
% OUTPUT-NEXT:4
% OUTPUT-NEXT:3
