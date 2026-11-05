% target data keeps arrays on the device for a goal, as Flang lowers !$omp target
% data: omp.target_data holds the goal, whose target regions find the arrays
% there, and what the goal binds leaves the region through slots. Without a device
% it maps nothing, and as comments it is the program's meaning.
%
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines

:- module omp_target_data.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

:- pred fill(int::in, float::in, array(float)::array_di, array(float)::array_uo)
    is det.
fill(N, X, !A) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !A ^ unsafe_elem(I) := X
    ).

:- pred add_to(int::in, array(float)::array_ui, array(float)::array_di,
    array(float)::array_uo) is det.
add_to(N, B, !A) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !A ^ unsafe_elem(I) := (!.A ^ unsafe_elem(I)) + (B ^ unsafe_elem(I))
    ).

:- func total(int, array(float)) = float.
:- mode total(in, array_ui) = out is det.
total(N, A) = Sum :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        S1 = S0 + A ^ unsafe_elem(I)
    ).

% The default map type, and kernels through calls, around a goal that is
% not on the device. A value it binds leaves the region.
% CHECK-LABEL: merk.proc private @omp_target_data.calls_4_0
% CHECK: omp.map.info {{.*}}map_clauses(tofrom) capture(ByRef) bounds({{.*}}) name("A")
% CHECK: omp.target_data map_entries({{.*}}) {
% CHECK: merk.call @omp_target_data.fill_4_0
% CHECK: llvm.store
% CHECK: omp.terminator
% CHECK: merk.throw_if {{.*}}"Software Error: target data: the array A was replaced in the region, not updated in place"
:- pred calls(int::in, float::out, io::di, io::uo) is det.
calls(N, Sum, !IO) :-
    some [!A] (
        array.init(N, 0.0, !:A),
        %$omp target data map(A)
        (
            fill(N, 1.5, !A),
            io.write_string("in the region\n", !IO),
            add_to(N, !.A, !A),
            Sum0 = total(N, !.A)
        ),
        % After the region, the host has the device's elements.
        Sum = Sum0 + (!.A ^ elem(1))
    ).

% A kernel in the region itself, on a state variable and an array only read.
% CHECK-LABEL: merk.proc private @omp_target_data.direct_3_0
% CHECK: omp.target_data map_entries(
% CHECK: omp.target
:- pred direct(int::in, array(float)::array_ui, float::out) is det.
direct(N, B, X) :-
    some [!A] (
        array.init(N, 0.0, !:A),
        %$omp target data map(tofrom: A) map(to: B)
        ( for(I, 0, N - 1) do
            %$omp target
            !A ^ unsafe_elem(I) := B ^ unsafe_elem(I)
        ),
        X = !.A ^ elem(0)
    ).

% Regions in regions.
:- pred nested(int::in, float::out) is det.
nested(N, X) :-
    some [!A, !B] (
        array.init(N, 1.0, !:A),
        array.init(N, 2.0, !:B),
        %$omp target data map(tofrom: A)
        (
            %$omp target data map(tofrom: B)
            add_to(N, !.A, !B),
            add_to(N, !.B, !A)
        ),
        X = (!.A ^ elem(0)) + (!.B ^ elem(0))
    ).

% from: nothing is copied to the device, and the goal sets every element.
:- pred from_only(int::in, float::out) is det.
from_only(N, X) :-
    some [!A] (
        array.init(N, 0.0, !:A),
        %$omp target data map(from: A)
        fill(N, 7.0, !A),
        X = !.A ^ elem(N - 1)
    ).


main(!IO) :-
    % OUTPUT:in the region
    calls(1000, Sum, !IO),
    % OUTPUT-NEXT:3003.0
    io.write_float(Sum, !IO), io.nl(!IO),
    array.init(16, 4.0, B),
    direct(16, B, X),
    % OUTPUT-NEXT:4.0
    io.write_float(X, !IO), io.nl(!IO),
    nested(8, Y),
    % OUTPUT-NEXT:7.0
    io.write_float(Y, !IO), io.nl(!IO),
    from_only(8, Z),
    % OUTPUT-NEXT:7.0
    io.write_float(Z, !IO), io.nl(!IO).
