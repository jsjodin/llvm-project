% The device's cells come from its arena (docs/INPROGRESS.md, The device's
% side of the first slice, step 6): bumped from chunks that the host
% allocates on the device when the device asks. Before each region the host
% gives it a slot, which the region passes to the device, whose cells are
% bumped from the slot's chunks; after it, the host frees them. Here three regions in turn each build
% a list longer than a chunk on one thread, and a loop's threads each
% build short lists, all at once, through many chunks. On the host the
% calls are made too, and nothing is allocated in an arena.
%
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_arena.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

% The module's own list functions, so that the arena is all this tests.
:- func build(int, list(int)) = list(int).
build(N, Xs) = ( if N = 0 then Xs else build(N - 1, [N | Xs]) ).

:- func sum(list(int), int) = int.
sum([], S) = S.
sum([X | Xs], S) = sum(Xs, S + X).

:- pred rounds(int::in, io::di, io::uo) is det.
rounds(R, !IO) :-
    ( if R > 0 then
        %$omp target
        Sum = sum(build(100000, []), 0),
        io.write_int(Sum, !IO), io.nl(!IO),
        rounds(R - 1, !IO)
    else
        true
    ).
% MERK-LABEL: merk.proc private @omp_target_arena.rounds_3_0(
% MERK:         %[[SLOT:[0-9]+]] = merk.foreign_call @merk_region_enter() : () -> i64
% MERK:         omp.target
% MERK:           merk.foreign_call @merk_device_region_enter(
% MERK:         merk.foreign_call @merk_region_exit(%[[SLOT]]) : (i64) -> ()

:- pred short_lists(int::in, int::out) is det.
short_lists(N, Sum) :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 1, N), fromto(0, S0, S1, Sum) do
        S1 = S0 + sum(build(I mod 8, []), 0)
    ).

main(!IO) :-
    rounds(3, !IO),
    short_lists(100000, Sum),
    io.write_int(Sum, !IO), io.nl(!IO).

% OUTPUT:5000050000
% OUTPUT-NEXT:5000050000
% OUTPUT-NEXT:5000050000
% OUTPUT-NEXT:1050000
