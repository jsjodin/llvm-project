% A reduction by /\ under %$omp parallel do (design document, "Loops",
% Under a parallel directive): its declaration starts each thread at -1,
% all bits set, and combines with llvm.and, as Flang's iand_i64 does.
% The program gives the same output without -fopenmp.
%
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-mlir %s | FileCheck %s --check-prefix=MLIR
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env OMP_NUM_THREADS=4 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t.O2 && %libomp-env OMP_NUM_THREADS=4 %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.seq && %t.seq | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module omp_parallel_do_reduction_and.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

main(!IO) :-
    common(0xff_ff, C), io.write_int(C, !IO), io.nl(!IO),
    common(0x0f_0f, D), io.write_int(D, !IO), io.nl(!IO).
% OUTPUT:65280
% OUTPUT-NEXT:3840

    % The bits that every one of 1000 numbers, each Mask with bit I mod 8
    % cleared, has: Mask's, but for bits 0 to 7.
:- pred common(int::in, int::out) is det.
common(Mask, A) :-
    %$omp parallel do reduction(/\)
    ( for(I, 0, 999), fromto(-1, A0, A1, A) do
        A1 = A0 /\ Mask /\ \ (1 << (I mod 8))
    ).
% MLIR-LABEL: omp.declare_reduction @iand_i64 : i64 init {
% MLIR:         llvm.mlir.constant(-1 : i64) : i64
% MLIR:       } combiner {
% MLIR:         llvm.and
