% Arrays cross into a target region as ClangIR maps a pointer's section,
% p[0:n]: by the array's own pointer, with bounds over its size word and
% its elements, which only the host has, to copy them. An array the region
% reads is mapped `to`; one it gives, updated in place, `tofrom`, and after
% the region the variable is the given array. Where the region's own
% operations do not show which array it gives, as when it calls a
% procedure that updates one, the region finds which of the given arrays
% of its type it is by its pointer, and gives its number, and the pointer
% too: on the host, a procedure the region calls may have allocated it.
% The output is mmc's for the same program, without the directives.
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp -emit-mlir -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MLIR
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-merk %s | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o /dev/null
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

% The squares region updates A0 through a call: the only given array of
% its type, mapped tofrom over its 11 words, 0 to its size.
% MERK-LABEL: merk.proc @omp_target_arrays.main_2_0()
% MERK: %[[A0:.*]] = merk.array_new
% MERK: %[[SIZE:.*]] = merk.array_size %[[A0]] : <i64>
% MERK: %[[EXTENT:.*]] = arith.addi %[[SIZE]], %{{.*}} : i64
% MERK: %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : i64) upper_bound(%[[SIZE]] : i64) extent(%[[EXTENT]] : i64)
% MERK-NEXT: %[[P:.*]] = builtin.unrealized_conversion_cast %[[A0]] : !merk.array<i64> to !llvm.ptr
% MERK-NEXT: omp.map.info var_ptr(%[[P]] : !llvm.ptr, i64) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) name("A0")
% Its number, and its pointer.
% MERK: omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i64) map_clauses(from) capture(ByRef) name("A1")
% MERK: omp.map.info var_ptr(%{{.*}} : !llvm.ptr, !llvm.ptr) map_clauses(from) capture(ByRef) name("A1")
% MERK: omp.target
% MERK: builtin.unrealized_conversion_cast %{{.*}} : !llvm.ptr to !merk.array<i64>
% MERK: merk.call @omp_target_arrays.squares_3_0(
% MERK: llvm.icmp "eq"
% MERK: omp.terminator
% The sum region only reads A1.
% MERK: omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i64) map_clauses(to) capture(ByRef) bounds(%{{.*}}) name("A1")
% The last region's own set shows that it gives F1: no number.
% MERK: omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i64) map_clauses(tofrom) capture(ByRef) bounds(%{{.*}}) name("F0")
% MERK: omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i64) map_clauses(tofrom) capture(ByRef) bounds(%{{.*}}) name("F1")
% MERK-NOT: name("F")
% MERK: omp.target kernel_type(generic) map_entries(%{{.*}} -> %arg0, %{{.*}} -> %arg1 : !llvm.ptr, !llvm.ptr)
% MERK: merk.array_set

% MLIR-LABEL: llvm.func tailcc @omp_target_arrays.main_2_0()
% MLIR: omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i64) map_clauses(tofrom) capture(ByRef) bounds(%{{.*}}) name("A0")
% MLIR-NOT: unrealized_conversion_cast

% The device has no bounds.
% DEVICE-LABEL: merk.proc @omp_target_arrays.main_2_0()
% DEVICE-NOT: omp.map.bounds
% DEVICE: omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i64) map_clauses(tofrom) capture(ByRef) name("A0")

% OUTPUT:285
% OUTPUT-NEXT:81
% OUTPUT-NEXT:0.25
% OUTPUT-NEXT:3.0
% OUTPUT-NEXT:a 3 0
% OUTPUT-NEXT:b 0 4
% OUTPUT-NEXT:a 0 8
% OUTPUT-NEXT:b 7 0
% OUTPUT-NEXT:r 9 8
% OUTPUT-NEXT:q 1 2
% OUTPUT-NOT:{{.}}

:- module omp_target_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, float.

:- pred sum(array(int)::array_ui, int::in, int::in, int::out) is det.
sum(A, I, S0, S) :-
    ( if I < array.size(A) then
        sum(A, I + 1, S0 + A ^ unsafe_elem(I), S)
    else
        S = S0
    ).

:- pred squares(int::in, array(int)::array_di, array(int)::array_uo) is det.
squares(I, !A) :-
    ( if I < array.size(!.A) then
        array.set(I, I * I, !A),
        squares(I + 1, !A)
    else
        true
    ).

:- pred scale(float::in, int::in, array(float)::array_di,
    array(float)::array_uo) is det.
scale(K, I, !A) :-
    ( if I < array.size(!.A) then
        array.lookup(!.A, I, X),
        array.set(I, K * X, !A),
        scale(K, I + 1, !A)
    else
        true
    ).

% Two given arrays of a type: the region finds which each output is.
:- pred both(int::in, array(int)::array_di, array(int)::array_di,
    array(int)::array_uo, array(int)::array_uo) is det.
both(N, A0, B0, A, B) :-
    array.set(0, N, A0, A),
    array.set(1, N + 1, B0, B).

:- pred swapped(int::in, array(int)::array_di, array(int)::array_di,
    array(int)::array_uo, array(int)::array_uo) is det.
swapped(N, A0, B0, B, A) :-
    array.set(0, N, A0, A),
    array.set(1, N + 1, B0, B).

:- pred one(int::in, array(int)::array_di, array(int)::array_di,
    array(int)::array_uo) is det.
one(N, A0, B0, R) :-
    ( if N > 5 then
        array.set(0, N, A0, R)
    else
        array.set(0, N, B0, R)
    ).

:- pred show(string::in, array(int)::array_ui, io::di, io::uo) is det.
show(Name, A, !IO) :-
    io.write_string(Name, !IO),
    io.write_int(A ^ elem(0), !IO), io.write_string(" ", !IO),
    io.write_int(A ^ elem(1), !IO), io.nl(!IO).

main(!IO) :-
    A0 = array.init(10, 0),
    %$omp target
    squares(0, A0, A1),
    %$omp target
    sum(A1, 0, 0, S),
    io.write_int(S, !IO), io.nl(!IO),
    io.write_int(A1 ^ elem(9), !IO), io.nl(!IO),
    F0 = array.init(3, 1.5),
    %$omp target
    scale(2.0, 0, F0, F1),
    %$omp target
    array.set(0, 0.25, F1, F),
    io.write_float(F ^ elem(0), !IO), io.nl(!IO),
    io.write_float(F ^ elem(2), !IO), io.nl(!IO),
    P0 = array.init(2, 0), Q0 = array.init(2, 0),
    %$omp target
    both(3, P0, Q0, P1, Q1),
    show("a ", P1, !IO), show("b ", Q1, !IO),
    %$omp target
    swapped(7, P1, Q1, P2, Q2),
    show("a ", P2, !IO), show("b ", Q2, !IO),
    %$omp target
    one(9, P2, Q2, R),
    show("r ", R, !IO),
    C0 = array.init(2, 1), D0 = array.init(2, 2),
    %$omp target
    one(1, C0, D0, T),
    show("q ", T, !IO).
