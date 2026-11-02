% Count to N: an if-then-else whose else part calls the procedure again as
% its last goal. The if-then-else is the procedure's last goal, so each
% branch succeeds on its own, and the recursive call is a tail call: it
% runs in constant stack space.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% MERK: merk.proc private @count.count_4_0(%[[I:.*]]: i64, %[[N:.*]]: i64) attributes {determinism = #merk.determinism<det>} {
% MERK-NEXT: merk.ite {
% MERK-NEXT: %[[GT:.*]] = arith.cmpi sgt, %[[I]], %[[N]] : i64
% MERK-NEXT: merk.guard %[[GT]]
% MERK-NEXT: merk.yield
% MERK-NEXT: } then {
% MERK-NEXT: merk.yield
% MERK-NEXT: } else {
% MERK-NEXT: merk.call @io.write_int_3_0(%[[I]]) : (i64) -> ()
% MERK-NEXT: merk.call @io.nl_2_0() : () -> ()
% MERK-NEXT: %[[ONE:.*]] = merk.constant 1 : i64
% MERK-NEXT: %[[NEXT:.*]] = arith.addi %[[I]], %[[ONE]] : i64
% MERK-NEXT: merk.call @count.count_4_0(%[[NEXT]], %[[N]]) : (i64, i64) -> ()
% MERK-NEXT: merk.yield
% MERK-NEXT: }
% MERK-NEXT: merk.succeed

% LLVM: define internal tailcc void @count.count_4_0(i64 %[[I:.*]], i64 %[[N:.*]]) {
% LLVM: %[[GT:.*]] = icmp sgt i64 %[[I]], %[[N]]
% LLVM-NEXT: br i1 %[[GT]], label %{{.*}}, label %[[ELSE:.*]]
% LLVM: ret void
% LLVM: [[ELSE]]:
% LLVM-NEXT: call tailcc void @io.write_int_3_0(i64 %[[I]])
% LLVM-NEXT: call tailcc void @io.nl_2_0()
% LLVM-NEXT: %[[NEXT:.*]] = add i64 %[[I]], 1
% LLVM-NEXT: musttail call tailcc void @count.count_4_0(i64 %[[NEXT]], i64 %[[N]])
% LLVM-NEXT: ret void

% OUTPUT:1
% OUTPUT-NEXT:2
% OUTPUT-NEXT:3
% OUTPUT-NEXT:4
% OUTPUT-NEXT:5
% OUTPUT-NEXT:6
% OUTPUT-NEXT:7
% OUTPUT-NEXT:8
% OUTPUT-NEXT:9
% OUTPUT-NEXT:10
% OUTPUT-NOT:{{.}}

:- module count.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.
:- pred count(int::in, int::in, io::di, io::uo) is det.
count(I, N, !IO) :-
    ( if I > N then
        true
    else
        io.write_int(I, !IO),
        io.nl(!IO),
        count(I + 1, N, !IO)
    ).
main(!IO) :-
    count(1, 10, !IO).
