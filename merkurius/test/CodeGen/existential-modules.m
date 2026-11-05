% An existentially typed predicate and function of another module, and a call in
% tail position that gives what its callee gave. The output is mmc's.
%
% RUN: merkc -I %S/Inputs %S/Inputs/exgive.m %s -o %t && %t | FileCheck %s --match-full-lines
% RUN: merkc -fc1 -emit-llvm -I %S/Inputs %s -o - | FileCheck %s --check-prefix=LLVM

:- module existential_modules.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module exgive, list.
:- pred u(T::in, io::di, io::uo) is det <= sh(T).
u(X, !IO) :- io.write_int(sz(X), !IO), io.write(X, !IO), io.nl(!IO).
% In tail position: it passes on what the callee gave, as the callee's results.
:- some [T] pred deep(int::in, T::out) is det => sh(T).
deep(N, X) :- give(N, X).
main(!IO) :-
    give(3, V), u(V, !IO), u(V, !IO),
    io.write(any(5), !IO), io.nl(!IO),
    deep(3000000, D), u(D, !IO).

% LLVM: define internal tailcc { ptr, i64 } @existential_modules.deep_2_0(i64 %0)
% LLVM-NEXT: musttail call tailcc { ptr, i64 } @exgive.give_2_0(i64 %0)
% CHECK: 43
% CHECK-NEXT: 43
% CHECK-NEXT: 10
% CHECK-NEXT: 30000013000000
