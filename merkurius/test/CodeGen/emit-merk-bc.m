% -emit-merk-bc writes the merk module as MLIR bytecode, whole: compiled
% from it, the module is the same as compiled from the source, with the
% collector's notes on roots that -emit-merk leaves out, its closures and its
% foreign code. The library is written so in a device's flavour, for a
% device's compile to copy procedures from (design document, Library code
% on the device).
%
% RUN: merkc -fc1 -emit-merk-bc -triple x86_64-unknown-linux-gnu %s -o %t.merkbc
% RUN: merkc -fc1 -emit-llvm %t.merkbc -o - | grep -v -e "^; ModuleID" -e "^source_filename" > %t.from-bytecode.ll
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | grep -v -e "^; ModuleID" -e "^source_filename" > %t.from-source.ll
% RUN: diff %t.from-source.ll %t.from-bytecode.ll
% RUN: FileCheck %s --check-prefix=LLVM < %t.from-bytecode.ll
%
% As for a device: the same merk module as from the source.
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-merk-bc %s -o %t.device.merkbc
% RUN: merkc -fc1 -emit-merk %t.device.merkbc -o %t.device-from-bytecode.merk
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-merk %s -o %t.device-from-source.merk
% RUN: diff %t.device-from-source.merk %t.device-from-bytecode.merk
% RUN: FileCheck %s --check-prefix=DEVICE < %t.device-from-bytecode.merk
%
% The text form reads back by its extension, .merk, and -x mlir reads
% bytecode whatever the file is called.
% RUN: merkc -fc1 -emit-llvm %t.device-from-source.merk -o /dev/null
% RUN: cp %t.merkbc %t.bytes
% RUN: merkc -fc1 -x mlir -emit-merk %t.bytes | FileCheck %s --check-prefix=ANY-NAME
% REQUIRES: amdgpu-registered-target

:- module emit_merk_bc.
:- interface.
:- import_module list.
:- func twice(list(T)) = list(T).
:- func add_all(int, list(int)) = list(int).
:- func plus_one(int) = int.
:- implementation.
:- import_module int.

twice(L) = L ++ L.

add_all(N, L) = list.map((func(X) = X + N), L).

:- pragma foreign_proc("C", plus_one(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe], "Y = X + 1;").

% LLVM-DAG: define {{.*}}@emit_merk_bc.twice_f1_0(
% LLVM-DAG: define {{.*}}@emit_merk_bc.add_all_f2_0(
% LLVM-DAG: define {{.*}}@"emit_merk_bc.add_all_f2_0$lambda0$closure1"(
% LLVM-DAG: define internal i64 @merk_foreign_emit__merk__bc_Dplus__one__f1__0(

% DEVICE: module @emit_merk_bc attributes {{{.*}}merk.foreign_code = {{.*}}Y = X + 1;
% DEVICE-DAG: merk.proc @emit_merk_bc.twice_f1_0(%arg0: !merk.type_info, %arg1: !merk.term<"list.list(T0)">)
% DEVICE-DAG: merk.closure

% ANY-NAME: merk.proc @emit_merk_bc.twice_f1_0(
