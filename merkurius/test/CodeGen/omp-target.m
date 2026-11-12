% %$omp target runs a goal on an OpenMP device, as an omp.target. Its
% inputs, the values it uses from outside, are mapped `to`, by copy; the
% variables it binds are mapped `from`. Both go through a stack slot of the
% procedure, as Flang maps scalars. The procedures it calls become declare
% target. The device compile keeps only those and the target region; there
% procedures use the C calling convention, since AMDGPU has no tailcc, and a
% failed check traps.
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s -o %t.mlir
% RUN: FileCheck %s --check-prefix=MERK < %t.mlir
% RUN: merkc -fc1 -fopenmp -emit-mlir -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MLIR
% The merk dump lowers as the module would have.
% RUN: merkc -fc1 -emit-mlir %t.mlir | FileCheck %s --check-prefix=MLIR
% The jobs the driver runs to offload (test/Driver/offload.test), one by
% one; -fopenmp-targets= is Clang's alias of --offload-targets=.
% RUN: merkc -fc1 -fopenmp -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=HOST
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-mlir %s | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s | FileCheck %s --check-prefix=DEVICE-LLVM
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm-bc %s -o %t.device.bc
% RUN: llvm-offload-binary -o %t.offload --image=file=%t.device.bc,triple=amdgpu10.30-amd-amdhsa,arch=gfx1030,kind=openmp
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -fembed-offload-object=%t.offload -emit-llvm -triple x86_64-unknown-linux-gnu -x ir %t.host.bc | FileCheck %s --check-prefix=EMBED
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -fembed-offload-object=%t.offload -emit-obj -triple x86_64-unknown-linux-gnu -x ir %t.host.bc -o %t.host.o
% RUN: llvm-objdump -h %t.host.o | FileCheck %s --check-prefix=SECTION
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int.

% A loop by tail recursion, which the target region calls.
:- pred sum(int::in, int::in, int::in, int::out) is det.
sum(I, N, Acc, S) :-
    ( if I > N then S = Acc else sum(I + 1, N, Acc + I * I, S) ).

:- pred compute(int::in, int::in, float::in, int::out, int::out, float::out)
    is det.
compute(N, D, X, S, Q, Y) :-
    %$omp target
    (
        sum(1, N, 0, S),
        Q = S // D,
        Y = X * 2.5 + float(N)
    ).

% An output may be an input's value.
:- pred same(int::in, int::out, int::out) is det.
same(X, Y, Z) :-
    %$omp target
    ( Y = X, Z = X + 1 ).

main(!IO) :-
    compute(100, 7, 1.5, S, Q, Y),
    io.write_int(S, !IO), io.nl(!IO),
    io.write_int(Q, !IO), io.nl(!IO),
    io.write_float(Y, !IO), io.nl(!IO),
    same(4, A, B),
    io.write_int(A, !IO), io.write_string(" ", !IO),
    io.write_int(B, !IO), io.nl(!IO).

% The module has OpenMP's attributes, and the goal is an omp.target, a
% generic kernel, whose map entries are its inputs, by copy, and its
% outputs. The first input is the slot that the host gives the region, which
% the region passes to the device before it makes a cell. Before it, the
% host checks that GPU processes leave the default device units for it.
% MERK: omp.is_gpu = false, omp.is_target_device = false, omp.target_triples = [], omp.version = #omp.version<version = 52>
% MERK-LABEL: merk.proc private @omp_target.compute_6_0(
% MERK-SAME: %[[N:arg0]]: i64, %[[D:arg1]]: i64, %[[X:arg2]]: f64
% MERK: %[[DEFAULT:[^ ]+]] = arith.constant -1 : i64
% MERK-NEXT: merk.foreign_call @merk_check_region_units(%[[DEFAULT]]) : (i64) -> ()
% MERK-NEXT: %[[SLOT:[0-9]+]] = merk.foreign_call @merk_region_enter() : () -> i64
% MERK-NEXT: llvm.store %[[SLOT]], %[[SLOTS:[0-9]+]] : i64, !llvm.ptr
% MERK-NEXT: %[[SLOTM:[0-9]+]] = omp.map.info var_ptr(%[[SLOTS]] : !llvm.ptr, i64) map_clauses(to) capture(ByCopy) name("region_slot")
% MERK: llvm.store %[[N]], %[[NS:[0-9]+]] : i64, !llvm.ptr
% MERK: %[[NM:[0-9]+]] = omp.map.info var_ptr(%[[NS]] : !llvm.ptr, i64) map_clauses(to) capture(ByCopy) name("N")
% MERK: omp.map.info {{.*}} map_clauses(to) capture(ByCopy) name("D")
% MERK: omp.map.info {{.*}} map_clauses(to) capture(ByCopy) name("X")
% MERK: %[[SM:[0-9]+]] = omp.map.info var_ptr(%[[SS:[0-9]+]] : !llvm.ptr, i64) map_clauses(from) capture(ByRef) name("S")
% MERK: omp.map.info {{.*}} map_clauses(from) capture(ByRef) name("Q")
% MERK: omp.map.info {{.*}} map_clauses(from) capture(ByRef) name("Y")
% MERK-NOT: omp.map.info
% MERK: omp.target kernel_type(generic) map_entries(%[[SLOTM]] -> %[[SLOTA:arg[0-9]+]], %[[NM]] -> %[[NA:arg[0-9]+]], {{.*}}, %[[SM]] -> %[[SA:arg[0-9]+]], {{.*}}) {
% MERK-NEXT: %[[SLOTV:[0-9]+]] = llvm.load %[[SLOTA]] : !llvm.ptr -> i64
% MERK-NEXT: %[[NV:[0-9]+]] = llvm.load %[[NA]] : !llvm.ptr -> i64
% MERK: merk.foreign_call @merk_device_region_enter(%[[SLOTV]]) : (i64) -> ()
% MERK: %[[S:[0-9]+]] = merk.call @omp_target.sum_4_0(%{{[0-9]+}}, %[[NV]], %{{[0-9]+}})
% MERK: merk.throw_if
% MERK: llvm.store %[[S]], %[[SA]] : i64, !llvm.ptr
% MERK: omp.terminator
% MERK: merk.foreign_call @merk_region_exit(%[[SLOT]]) : (i64) -> ()
% MERK-NEXT: llvm.load %[[SS]] : !llvm.ptr -> i64
% MERK-LABEL: merk.proc private @omp_target.same_3_0(
% MERK: omp.target kernel_type(generic) map_entries(%{{[0-9]+}} -> %{{arg[0-9]+}}, %{{[0-9]+}} -> %[[XA:arg[0-9]+]], %{{[0-9]+}} -> %[[YA:arg[0-9]+]], %{{[0-9]+}} -> %{{arg[0-9]+}} :
% MERK-NEXT: llvm.load %{{arg[0-9]+}} : !llvm.ptr -> i64
% MERK-NEXT: %[[XV:[0-9]+]] = llvm.load %[[XA]] : !llvm.ptr -> i64
% MERK: llvm.store %[[XV]], %[[YA]] : i64, !llvm.ptr

% Lowered, the region's goals are blocks, and what it calls is declare
% target.
% MLIR-LABEL: llvm.func internal tailcc @omp_target.compute_6_0(
% MLIR: omp.target kernel_type(generic)
% MLIR: llvm.call tailcc @omp_target.sum_4_0(
% MLIR: llvm.cond_br
% MLIR: llvm.call @merk_uncaught_exception(
% MLIR: omp.terminator
% MLIR-LABEL: llvm.func internal tailcc @omp_target.sum_4_0(
% A procedure only target regions call, and private to the module, is
% nohost, as omp-mark-declare-target marks Flang's: the host still has it,
% for running the region there.
% MLIR-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to, implicit = true>

% With a device triple, the host makes the region an offload entry, which
% it launches.
% HOST: @.offloading.entry.__omp_offloading_{{[0-9a-f]+}}_{{[0-9a-f]+}}_omp_target.compute_6_0_l{{[0-9]+}} = weak constant %struct.__tgt_offload_entry
% HOST-LABEL: define internal tailcc { i64, i64, double } @omp_target.compute_6_0(
% HOST: call i32 @__tgt_target_kernel(

% The device keeps the target region and the procedures it calls, with
% the C convention; the host's slots become arguments.
% The AMDGPU target info's code object version, as ClangIR records it.
% DEVICE: merk.amdgpu_code_object_version = 600 : i32
% DEVICE-SAME: omp.is_gpu = true, omp.is_target_device = true
% DEVICE-NOT: @omp_target.main_2_0
% DEVICE-LABEL: llvm.func internal @omp_target.compute_6_0(
% DEVICE-SAME: !llvm.ptr<5>
% DEVICE: llvm.addrspacecast
% DEVICE: omp.target kernel_type(generic)
% DEVICE: llvm.call @omp_target.sum_4_0(
% DEVICE-LABEL: llvm.func internal @omp_target.sum_4_0(
% DEVICE: llvm.call musttail @omp_target.sum_4_0(
% DEVICE-NOT: merk_main
% DEVICE-NOT: merk_uncaught_exception

% DEVICE-LLVM: target triple = "amdgpu10.30-amd-amdhsa"
% DEVICE-LLVM-LABEL: define internal i64 @omp_target.sum_4_0(
% DEVICE-LLVM: musttail call i64 @omp_target.sum_4_0(
% DEVICE-LLVM-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{[0-9a-f]+}}_{{[0-9a-f]+}}_omp_target.compute_6_0_l{{[0-9]+}}(
% DEVICE-LLVM: call void @llvm.trap()
% DEVICE-LLVM-NOT: merk_uncaught_exception
% DEVICE-LLVM-DAG: !{i32 1, !"amdhsa_code_object_version", i32 600}
% DEVICE-LLVM-DAG: !{i32 8, !"PIC Level", i32 2}

% The host's IR read back keeps its kernel launch, and gets the offload
% binary in .llvm.offloading, for the linker wrapper.
% EMBED: @llvm.embedded.object = private constant [{{[0-9]+}} x i8] c"\10\FF\10\AD{{.*}}", section ".llvm.offloading", align 8, !exclude
% EMBED: call i32 @__tgt_target_kernel(
% SECTION: .llvm.offloading

% Without a device, the region runs on the host.
% OUTPUT:338350
% OUTPUT-NEXT:48335
% OUTPUT-NEXT:103.75
% OUTPUT-NEXT:4 5
