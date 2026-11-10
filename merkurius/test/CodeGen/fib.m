% Fibonacci numbers, the first 21: a doubly recursive function, and a loop
% whose then part ends with a tail call.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.O2 && %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% MERK: merk.proc private @fib.fib_f1_0(%[[N:.*]]: i64) -> i64 attributes {determinism = #merk.determinism<det>} {
% MERK-NEXT: %[[F:.*]] = merk.ite -> i64 {
% MERK: arith.cmpi slt, %[[N]], %{{.*}} : i64
% MERK: } then {
% MERK-NEXT: merk.yield %[[N]] : i64
% MERK-NEXT: } else {
% MERK: %[[A:.*]] = merk.call @fib.fib_f1_0(%{{.*}}) : (i64) -> i64
% MERK: %[[B:.*]] = merk.call @fib.fib_f1_0(%{{.*}}) : (i64) -> i64
% MERK-NEXT: %[[S:.*]] = arith.addi %[[A]], %[[B]] : i64
% MERK-NEXT: merk.yield %[[S]] : i64
% MERK-NEXT: }
% MERK-NEXT: merk.succeed %[[F]] : i64

% LLVM: define internal tailcc void @fib.loop_3_0(i64 %[[I:.*]]) {
% LLVM: musttail call tailcc void @fib.loop_3_0(i64 %{{.*}})
% LLVM-NEXT: ret void
% LLVM: define internal tailcc i64 @fib.fib_f1_0(i64 %[[N:.*]]) {
% LLVM: ret i64 %[[N]]

% OUTPUT:0
% OUTPUT-NEXT:1
% OUTPUT-NEXT:1
% OUTPUT-NEXT:2
% OUTPUT-NEXT:3
% OUTPUT-NEXT:5
% OUTPUT-NEXT:8
% OUTPUT-NEXT:13
% OUTPUT-NEXT:21
% OUTPUT-NEXT:34
% OUTPUT-NEXT:55
% OUTPUT-NEXT:89
% OUTPUT-NEXT:144
% OUTPUT-NEXT:233
% OUTPUT-NEXT:377
% OUTPUT-NEXT:610
% OUTPUT-NEXT:987
% OUTPUT-NEXT:1597
% OUTPUT-NEXT:2584
% OUTPUT-NEXT:4181
% OUTPUT-NEXT:6765
% OUTPUT-NOT:{{.}}

:- module fib.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.
:- func fib(int) = int.
fib(N) = ( if N < 2 then N else fib(N - 1) + fib(N - 2) ).
:- pred loop(int::in, io::di, io::uo) is det.
loop(I, !IO) :-
    ( if I =< 20 then
        io.write_int(fib(I), !IO), io.nl(!IO), loop(I + 1, !IO)
    else
        true
    ).
main(!IO) :-
    loop(0, !IO).
