% The smallest programs: main/2 with nothing to do, and one that calls a
% procedure of its own. A value of type io.state is not passed, so the
% procedures take no arguments; merk.entry names main(di, uo), which the
% runtime calls through merk_main. main's call of step is its last goal, so
% it is a tail call.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-mlir -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MLIR
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM

% MERK-LABEL: module @main attributes
% MERK-SAME: llvm.target_triple = "x86_64-unknown-linux-gnu"
% MERK: merk.proc @main.main_2_0() attributes {determinism = #merk.determinism<det>} {
% MERK-NEXT: merk.call @main.step_2_0() : () -> ()
% MERK-NEXT: merk.succeed
% MERK: merk.proc private @main.step_2_0() attributes {determinism = #merk.determinism<det>} {
% MERK-NEXT: merk.succeed
% MERK: merk.entry @main.main_2_0

% MLIR: llvm.func tailcc @main.main_2_0() {
% MLIR-NEXT: llvm.call tailcc musttail @main.step_2_0() : () -> ()
% MLIR-NEXT: llvm.return
% MLIR: llvm.func internal tailcc @main.step_2_0() {
% MLIR: llvm.func @merk_main() {
% MLIR-NEXT: llvm.call tailcc @main.main_2_0() : () -> ()

% LLVM: target triple = "x86_64-unknown-linux-gnu"
% LLVM: define tailcc void @main.main_2_0() {
% LLVM-NEXT: musttail call tailcc void @main.step_2_0()
% LLVM-NEXT: ret void
% LLVM: define internal tailcc void @main.step_2_0() {
% LLVM: define void @merk_main() {

:- module main.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
main(!IO) :-
    step(!IO).
:- pred step(io::di, io::uo) is det.
step(!IO).
