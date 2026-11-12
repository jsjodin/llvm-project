% Narrow arrays cross into target regions as word arrays do, mapped by
% their words, the header's and the elements' (runtime/array.h), which
% merk.array_words gives: 13 uint8s take 2 words after the header, 5 int16s
% 2, 7 colours 4. A region updates a uint8 array in place, reads an int16
% array, paints an enumeration array, and makes a uint8 array of its own,
% which the device copies to the host as the region ends, by its header.
% The output is mmc's for the same program, without the directives.
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

% MERK-LABEL: merk.proc @omp_target_narrow_arrays.main_2_0()
% MERK: %[[WORDS:.*]] = merk.array_words %{{.*}} : <i8>
% MERK: omp.map.bounds lower_bound(%{{.*}} : i64) upper_bound(%{{.*}} : i64) extent(%[[WORDS]] : i64)
% MERK: name("U0")

% The device stores a uint8 as its byte, and makes an array 1 byte wide.
% DEVICE-LABEL: define {{.*}} @omp_target_narrow_arrays.bump_3_0(
% DEVICE: store i8
% DEVICE-LABEL: define {{.*}} @omp_target_narrow_arrays.made_3_0(
% DEVICE: call {{.*}} @merk_array_new(i64 %{{.*}}, i64 9, i64 1)

% OUTPUT:array([1u8, 2u8, 3u8, 4u8, 101u8, 128u8, 129u8, 200u8, 201u8, 254u8, 255u8, 0u8, 1u8])
% OUTPUT-NEXT:32397
% OUTPUT-NEXT:array([red, green, blue, red, green, blue, red])
% OUTPUT-NEXT:array([9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 200u8])
% OUTPUT-NEXT:array([5u8, 0u8])
% OUTPUT-NOT:{{.}}

:- module omp_target_narrow_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, int16, list, uint8.

:- type colour
    --->    red
    ;       green
    ;       blue.

    % Adds one to each element from I on, 255 going to 0.
:- pred bump(int::in, array(uint8)::array_di, array(uint8)::array_uo) is det.
bump(I, !A) :-
    ( if I < array.size(!.A) then
        array.set(I, !.A ^ elem(I) + 1u8, !A),
        bump(I + 1, !A)
    else
        true
    ).

:- pred sum16(array(int16)::array_ui, int::in, int::in, int::out) is det.
sum16(A, I, S0, S) :-
    ( if I < array.size(A) then
        sum16(A, I + 1, S0 + int16.to_int(A ^ elem(I)), S)
    else
        S = S0
    ).

    % Element I is red, green or blue, by I mod 3, from I on.
:- pred paint(int::in, array(colour)::array_di, array(colour)::array_uo)
    is det.
paint(I, !A) :-
    ( if I < array.size(!.A) then
        M = I mod 3,
        C = ( if M = 0 then red else if M = 1 then green else blue ),
        array.set(I, C, !A),
        paint(I + 1, !A)
    else
        true
    ).

    % A new array of N elements, the last 200 and the others 9, or A0 with
    % its first element 5.
:- pred made(int::in, array(uint8)::array_di, array(uint8)::array_uo) is det.
made(N, A0, A) :-
    ( if N > 2 then
        array.init(N, 9u8, A1),
        array.set(N - 1, 200u8, A1, A)
    else
        array.set(0, 5u8, A0, A)
    ).

main(!IO) :-
    U0 = array([0u8, 1u8, 2u8, 3u8, 100u8, 127u8, 128u8, 199u8, 200u8,
        253u8, 254u8, 255u8, 0u8]),
    %$omp target
    bump(0, U0, U1),
    io.write_line(U1, !IO),
    S0 = array([-300i16, 5i16, -7i16, 32767i16, -68i16]),
    %$omp target
    sum16(S0, 0, 0, S),
    io.write_int(S, !IO), io.nl(!IO),
    C0 = array.init(7, red),
    %$omp target
    paint(0, C0, C1),
    io.write_line(C1, !IO),
    X0 = array.init(2, 0u8),
    %$omp target
    made(11, X0, X),
    io.write_line(X, !IO),
    Y0 = array.init(2, 0u8),
    %$omp target
    made(1, Y0, Y),
    io.write_line(Y, !IO).
