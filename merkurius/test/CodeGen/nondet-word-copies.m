% Word copies of nondet and multi procedures: a closure's entry calls the
% procedure's word copy, which gives each solution's outputs as words, and
% so passes its continuation on as a tail call, as does a call in the copy
% or in a lambda expression whose solutions it only gives on, through a
% disjunction's merk.each too, whose continuation takes the outputs as
% words. Each loop below runs three million times in constant stack, at
% every optimization level, as in mmc. The output is mmc's for the same
% program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module nondet_word_copies.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module float, int, list, solutions.

% Each has three million solutions, or calls itself three million times
% before its first.
:- pred count_down(int::in, int::out) is nondet.
count_down(N, X) :-
    N > 0,
    ( X = N ; count_down(N - 1, X) ).

:- pred halves(int::in, float::out) is nondet.
halves(N, X) :-
    N > 0,
    ( X = float(N) / 2.0 ; halves(N - 1, X) ).

:- pred deep(int::in, int::out) is multi.
deep(N, X) :-
    ( if N = 0 then ( X = 7 ; X = 8 ) else deep(N - 1, X) ).

:- pred last(pred(T)::in(pred(out) is nondet), T::in, T::out) is cc_multi.
last(P, X0, X) :-
    unsorted_aggregate(P, (pred(Y::in, _::in, Z::out) is det :- Z = Y), X0, X).

:- pred show(T::in, io::di, io::uo) is det.
show(X, !IO) :- io.write(X, !IO), io.nl(!IO).

main(!IO) :-
    last(count_down(3000000), 0, A), show(A, !IO),
    last(halves(3000000), 0.0, B), show(B, !IO),
    solutions(deep(3000000), C), show(C, !IO),
    % A lambda expression whose disjunction calls a nondet procedure last.
    N = 3000000,
    P = (pred(X::out) is nondet :- ( X = -1 ; count_down(N, X) )),
    last(P, 0, D), show(D, !IO).

% MERK-LABEL: merk.proc private @nondet_word_copies.halves_2_0$words(%arg0: i64) -> !merk.word attributes {determinism = #merk.determinism<nondet>} {
% MERK: merk.each {
% MERK: %[[H:.*]] = arith.divf
% MERK-NEXT: %[[W:.*]] = merk.to_word %[[H]] : f64
% MERK-NEXT: merk.solution %[[W]] : !merk.word
% MERK: merk.nondet_call @nondet_word_copies.halves_2_0$words(%{{.*}} : i64) {
% MERK-NEXT: ^bb0(%[[X:.*]]: !merk.word):
% MERK-NEXT: merk.solution %[[X]] : !merk.word
% MERK-NEXT: }
% MERK-NEXT: }
% MERK-NEXT: } then {
% MERK-NEXT: ^bb0(%[[Y:.*]]: !merk.word):
% MERK-NEXT: merk.succeed %[[Y]] : !merk.word
% MERK-LABEL: merk.proc private @nondet_word_copies.halves_2_0$words$closure1(%arg0: !merk.closure) -> !merk.word attributes {determinism = #merk.determinism<nondet>} {
% MERK: merk.nondet_call @nondet_word_copies.halves_2_0$words(
% MERK-LABEL: merk.proc private @nondet_word_copies.main_2_0$lambda0(%arg0: i64) -> !merk.word attributes {determinism = #merk.determinism<nondet>} {
% MERK: merk.nondet_call @nondet_word_copies.count_down_2_0$words(%arg0 : i64) {

% LLVM-LABEL: define internal tailcc i1 @"nondet_word_copies.halves_2_0$words"(i64 %0, ptr %1, ptr %2)
% LLVM: call tailcc i1 %1(ptr %2, ptr addrspace(1) %{{.*}})
% LLVM: musttail call tailcc i1 @"nondet_word_copies.halves_2_0$words"(i64 %{{.*}}, ptr %1, ptr %2)
% LLVM-LABEL: define internal tailcc i1 @"nondet_word_copies.halves_2_0$words$closure1"(ptr addrspace(1) %0, ptr %1, ptr %2)
% LLVM: musttail call tailcc i1 @"nondet_word_copies.halves_2_0$words"(i64 %{{.*}}, ptr %1, ptr %2)
% LLVM-LABEL: define internal tailcc i1 @"nondet_word_copies.main_2_0$lambda0"(i64 %0, ptr %1, ptr %2)
% LLVM: musttail call tailcc i1 @"nondet_word_copies.count_down_2_0$words"(i64 %0, ptr %1, ptr %2)

% OUTPUT:1
% OUTPUT-NEXT:0.5
% OUTPUT-NEXT:[7, 8]
% OUTPUT-NEXT:1
