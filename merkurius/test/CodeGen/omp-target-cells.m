% A term crosses into a target region when its type has one constructor,
% of two or more scalars: its pointer has no tag, and its cell, a word for
% each argument, is mapped `to` by the pointer, with the cell as the
% section, as an array is. The device takes it apart as the host does.
% A ground term known at compile time, a static cell, crosses the same way.
% The output is mmc's for the same program, without the directive.
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp -emit-mlir -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MLIR
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-merk %s | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o /dev/null
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_cells.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int.

:- type point
    --->    point(int, int, float).

:- type pair
    --->    pair(int, int).

:- func make(int) = point.
make(N) = point(N, N * 2, float(N) / 4.0).

:- func area(point) = float.
area(point(X, Y, Scale)) = float(X * Y) * Scale.

main(!IO) :-
    P = make(3),
    Q = pair(7, 8),
    %$omp target
    (
        P = point(X, Y, _),
        Q = pair(A, B),
        S = X + Y + A * B,
        F = area(P)
    ),
    io.write_int(S, !IO),
    io.nl(!IO),
    io.write_float(F, !IO),
    io.nl(!IO).
% P's cell is three words, 0 to 2; Q's two.
% MERK-LABEL: merk.proc @omp_target_cells.main_2_0()
% MERK: %[[P:.*]] = merk.call @omp_target_cells.make_f1_0(
% MERK: %[[Q:.*]] = merk.construct #merk.functor<"pair", 2>
% MERK: %[[PB:.*]] = omp.map.bounds lower_bound(%{{.*}} : i64) upper_bound(%{{.*}} : i64) extent(%{{.*}} : i64)
% MERK-NEXT: %[[PP:.*]] = builtin.unrealized_conversion_cast %[[P]] : !merk.term<"omp_target_cells.point"> to !llvm.ptr
% MERK-NEXT: omp.map.info var_ptr(%[[PP]] : !llvm.ptr, i64) map_clauses(to) capture(ByRef) bounds(%[[PB]]) name("P")
% MERK: %[[QQ:.*]] = builtin.unrealized_conversion_cast %[[Q]] : !merk.term<"omp_target_cells.pair"> to !llvm.ptr
% MERK-NEXT: omp.map.info var_ptr(%[[QQ]] : !llvm.ptr, i64) map_clauses(to) capture(ByRef) bounds(%{{.*}}) name("Q")
% MERK: omp.target kernel_type(generic) map_entries(%{{[0-9]+}} -> %[[ARGP:[^,]*]], %{{[0-9]+}} -> %[[ARGQ:[^,]*]],
% MERK-NEXT: %[[TP:.*]] = builtin.unrealized_conversion_cast %[[ARGP]] : !llvm.ptr to !merk.term<"omp_target_cells.point">
% MERK-NEXT: %[[TQ:.*]] = builtin.unrealized_conversion_cast %[[ARGQ]] : !llvm.ptr to !merk.term<"omp_target_cells.pair">
% MERK-NEXT: merk.deconstruct #merk.functor<"point", 3> %[[TP]]
% MERK-NEXT: merk.deconstruct #merk.functor<"pair", 2> %[[TQ]]

% MLIR-LABEL: llvm.func tailcc @omp_target_cells.main_2_0()
% MLIR: omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i64) map_clauses(to) capture(ByRef) bounds(%{{.*}}) name("P")
% MLIR-NOT: unrealized_conversion_cast

% The device has no bounds.
% DEVICE-LABEL: merk.proc @omp_target_cells.main_2_0()
% DEVICE-NOT: omp.map.bounds
% DEVICE: omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i64) map_clauses(to) capture(ByRef) name("P")
% DEVICE: omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i64) map_clauses(to) capture(ByRef) name("Q")

% OUTPUT:65
% OUTPUT-NEXT:13.5
