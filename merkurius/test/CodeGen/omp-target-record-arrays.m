% Arrays of flat records cross into target regions as narrow arrays do,
% mapped by their words, the header's and the records' (runtime/array.h),
% which merk.array_words gives: a cell is { { i32, i32 }, i8, i32, double },
% 24 bytes as C lays it out. A region updates an array of cells in place,
% loading and storing each cell whole; one reads an array; and one makes an
% array of its own, which the device copies to the host as the region ends,
% by its header. The output is mmc's for the same program, without the
% directives.
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

% MERK-LABEL: merk.proc @omp_target_record_arrays.main_2_0()
% MERK: %[[WORDS:.*]] = merk.array_words %{{.*}} : <!merk.term<"omp_target_record_arrays.cell">>
% MERK: omp.map.bounds lower_bound(%{{.*}} : i64) upper_bound(%{{.*}} : i64) extent(%[[WORDS]] : i64)
% MERK: name("A0")

% The device reads and writes a cell in place, after the header's two words,
% and makes an array of 24-byte records.
% DEVICE-LABEL: define {{.*}} @omp_target_record_arrays.step_3_0(
% DEVICE: %[[AT:.*]] = getelementptr i64, ptr %{{.*}}, i32 2
% DEVICE: %[[CELL:.*]] = getelementptr { { i32, i32 }, i8, i32, double }, ptr %[[AT]], i64 %{{.*}}
% DEVICE: load { { i32, i32 }, i8, i32, double }, ptr %[[CELL]]
% DEVICE: store { { i32, i32 }, i8, i32, double } %{{.*}}, ptr %{{.*}}
% DEVICE-LABEL: define {{.*}} @omp_target_record_arrays.made_3_0(
% DEVICE: call {{.*}} @merk_array_new_bytes(i64 %{{.*}}, ptr %{{.*}}, i64 24)

% OUTPUT:array([cell(point(250i32, -250i32), 251u8, red, 0.0), cell(point(252i32, -249i32), 252u8, green, 0.5), cell(point(254i32, -248i32), 253u8, blue, 1.0), cell(point(256i32, -247i32), 254u8, red, 1.5), cell(point(258i32, -246i32), 255u8, green, 2.0), cell(point(260i32, -245i32), 0u8, blue, 2.5), cell(point(6i32, 12i32), 1u8, red, 3.0)])
% OUTPUT-NEXT:10.5
% OUTPUT-NEXT:array([cell(point(0i32, 0i32), 0u8, blue, 0.5), cell(point(1i32, 1i32), 1u8, blue, 0.5), cell(point(2i32, 2i32), 2u8, blue, 0.5), cell(point(3i32, 3i32), 3u8, blue, 0.5)])
% OUTPUT-NOT:{{.}}

:- module omp_target_record_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, int32, list, uint8.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type point ---> point(x :: int32, y :: int32).
:- type cell
    --->    cell(
                at :: point,
                count :: uint8,
                hue :: colour,
                heat :: float
            ).

    % Moves each cell from I on by its count, adds one to the count and
    % halves the heat.
:- pred step(int::in, array(cell)::array_di, array(cell)::array_uo) is det.
step(I, !A) :-
    ( if I < array.size(!.A) then
        C = !.A ^ elem(I),
        N = int32.cast_from_int(uint8.to_int(C ^ count)),
        P = point(C ^ at ^ x + N, C ^ at ^ y - N),
        array.set(I, cell(P, C ^ count + 1u8, C ^ hue, C ^ heat / 2.0), !A),
        step(I + 1, !A)
    else
        true
    ).

:- pred total(array(cell)::array_ui, int::in, float::in, float::out) is det.
total(A, I, S0, S) :-
    ( if I < array.size(A) then
        total(A, I + 1, S0 + A ^ elem(I) ^ heat, S)
    else
        S = S0
    ).

    % A new array of N cells, each blue at (I, I).
:- pred made(int::in, array(cell)::array_di, array(cell)::array_uo) is det.
made(N, A0, A) :-
    ( if N > 0 then
        array.init(N, cell(point(0i32, 0i32), 0u8, blue, 1.0), A1),
        fill(0, A1, A)
    else
        A = A0
    ).

:- pred fill(int::in, array(cell)::array_di, array(cell)::array_uo) is det.
fill(I, !A) :-
    ( if I < array.size(!.A) then
        J = int32.cast_from_int(I),
        array.set(I, cell(point(J, J), uint8.cast_from_int(I), blue, 0.5), !A),
        fill(I + 1, !A)
    else
        true
    ).

:- func colour_of(int) = colour.
colour_of(I) =
    ( if I mod 3 = 0 then red else if I mod 3 = 1 then green else blue ).

:- func start(int) = cell.
start(I) = cell(point(int32.cast_from_int(I), int32.cast_from_int(2 * I)),
    uint8.cast_from_int(250 + I), colour_of(I), float(I)).

main(!IO) :-
    A0 = array(list.map(start, 0 .. 6)),
    %$omp target
    step(0, A0, A1),
    %$omp target
    total(A1, 0, 0.0, S),
    io.write_line(A1, !IO),
    io.write_line(S, !IO),
    X0 = array.init(1, start(0)),
    %$omp target
    made(4, X0, X),
    io.write_line(X, !IO).
