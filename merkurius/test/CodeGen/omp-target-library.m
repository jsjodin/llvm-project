% The library's procedures on a device (design document, Library code on
% the device): the device's compile copies those its target region
% reaches from the library's merk bytecode, with the procedures they call,
% as internal functions, and the descriptor of maybe(T), whose type_info the
% region makes and which the module only declares.
%
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-merk %s -o - | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -O0 -fopenmp %libomp-flags %s -o %t0 && %libomp-env %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t2 && %libomp-env %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_library.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe.

main(!IO) :-
    %$omp target
    count(N),
    io.write_int(N, !IO),
    io.nl(!IO).

:- pred count(int::out) is det.

count(N) :-
    L = [yes(1), no, yes(3)],
    N = list.length(L) * 100 + list.length(list.reverse(L ++ L)).

% A copy is private, and says which procedure first reached it.
% MERK-DAG: merk.type_ctor "maybe.maybe/1" module "maybe" name "maybe" kind "term" linkage "linkonce" arity 1 term !merk.term<"maybe.maybe(T0)">
% MERK-DAG: merk.type_def "maybe.maybe(T0)"
% MERK-DAG: merk.proc private @list.length_f1_0({{.*}}merk.reached_from = "omp_target_library.count_1_0"
% MERK-DAG: merk.proc private @list.length_2_0({{.*}}merk.reached_from = "list.length_f1_0"
% MERK-DAG: merk.proc private @list.reverse_f1_0(
% MERK-DAG: merk.proc private @list.append_3_1(
% MERK-NOT: merk.type_ctor "maybe.maybe/1" {{.*}}linkage "declaration"

% DEVICE-DAG: define internal i64 @list.length_f1_0(
% DEVICE-DAG: define internal i64 @list.length_acc_3_0(
% DEVICE-DAG: define internal ptr @list.reverse_acc_3_0(
% DEVICE-DAG: @"merk.type_ctor.maybe.maybe/1" = {{.*}}constant
% DEVICE-NOT: merk.reached

% OUTPUT:306
% OUTPUT-NOT:{{.}}
