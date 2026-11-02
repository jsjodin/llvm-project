% Factorial: a function whose value is that of an if-then-else, the
% merk.ite's result. The recursive call is not the last goal, so it is not
% a tail call.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% MERK: merk.proc private @factorial.fact_f1_0(%[[N:.*]]: i64) -> i64 attributes {determinism = #merk.determinism<det>} {
% MERK-NEXT: %[[F:.*]] = merk.ite -> i64 {
% MERK-NEXT: %[[ZERO:.*]] = merk.constant 0 : i64
% MERK-NEXT: %[[LE:.*]] = arith.cmpi sle, %[[N]], %[[ZERO]] : i64
% MERK-NEXT: merk.guard %[[LE]]
% MERK-NEXT: merk.yield
% MERK-NEXT: } then {
% MERK-NEXT: %[[ONE:.*]] = merk.constant 1 : i64
% MERK-NEXT: merk.yield %[[ONE]] : i64
% MERK-NEXT: } else {
% MERK-NEXT: %[[ONE:.*]] = merk.constant 1 : i64
% MERK-NEXT: %[[M:.*]] = arith.subi %[[N]], %[[ONE]] : i64
% MERK-NEXT: %[[R:.*]] = merk.call @factorial.fact_f1_0(%[[M]]) : (i64) -> i64
% MERK-NEXT: %[[P:.*]] = arith.muli %[[N]], %[[R]] : i64
% MERK-NEXT: merk.yield %[[P]] : i64
% MERK-NEXT: }
% MERK-NEXT: merk.succeed %[[F]] : i64

% LLVM: define internal tailcc i64 @factorial.fact_f1_0(i64 %[[N:.*]]) {
% LLVM: %[[LE:.*]] = icmp sle i64 %[[N]], 0
% LLVM-NEXT: br i1 %[[LE]], label %{{.*}}, label %[[ELSE:.*]]
% LLVM: ret i64 1
% LLVM: [[ELSE]]:
% LLVM-NEXT: %[[M:.*]] = sub i64 %[[N]], 1
% LLVM-NEXT: %[[R:.*]] = call tailcc i64 @factorial.fact_f1_0(i64 %[[M]])
% LLVM-NEXT: %[[P:.*]] = mul i64 %[[N]], %[[R]]
% LLVM-NEXT: ret i64 %[[P]]

% OUTPUT:2432902008176640000
% OUTPUT-NOT:{{.}}

:- module factorial.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.
:- func fact(int) = int.
fact(N) = ( if N =< 0 then 1 else N * fact(N - 1) ).
main(!IO) :-
    io.write_int(fact(20), !IO), io.nl(!IO).
