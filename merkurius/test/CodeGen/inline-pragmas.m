% A predicate's or function's no_inline and inline pragmas reach LLVM as
% noinline and alwaysinline, since merkc has no inliner of its own: kept,
% small enough to inline, stays a call at -O2, and folded, too large to
% inline, has no function of its own, even at -O0, where LLVM's
% always-inliner still runs; the closure list.map calls has it inlined.
% kept's argument is not known, so that the call is not folded away. Before
% 2026-11-17 both pragmas were read and then ignored.
%
% RUN: merkc -fc1 -O2 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=IR --implicit-check-not='@inline_pragmas.folded_f1_0('
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=O0 --implicit-check-not='@inline_pragmas.folded_f1_0('
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t
% RUN: %t | FileCheck %s --match-full-lines --strict-whitespace
%
% IR: call tailcc i64 @inline_pragmas.kept_2_0(
% IR: define internal tailcc i64 @inline_pragmas.kept_2_0({{.*}} #[[KEPT:[0-9]+]] {
% IR: attributes #[[KEPT]] = { {{.*}}noinline
% O0: define internal tailcc i64 @inline_pragmas.kept_2_0({{.*}} #[[KEPT:[0-9]+]] {
% O0: attributes #[[KEPT]] = { noinline }
%
% CHECK:341652267210276
% CHECK-NEXT:[32189748307203, 115950922338624, 226766312972649]
% CHECK-NOT:{{.}}

:- module inline_pragmas.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    kept(list.length(Args) + 3, X),
    Y = folded(X),
    io.write_int(X + Y, !IO),
    io.nl(!IO),
    io.write_line(list.map(folded, [1, 2, 3]), !IO).

    % Small enough that LLVM would inline it.
:- pred kept(int::in, int::out) is det.
:- pragma no_inline(kept/2).

kept(X, X + 1).

    % Large enough that LLVM would not.
:- func folded(int) = int.
:- pragma inline(folded/1).

folded(X) = Y :-
    A = X * 3 + 1, B = A * A - X, C = B mod 7 + A * 5, D = C * C + B,
    E = D mod 11 + C * 13, F = E * E - D, G = F mod 17 + E * 19,
    H = G * G + F, I = H mod 23 + G * 29, J = I * I - H,
    K = J mod 31 + I * 37, Y = K * K + J mod 41.
