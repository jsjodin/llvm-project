% A descriptor's compare entry is not compiled for a device, so comparing
% values of a type variable there is an error, at the comparison.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -fdo-loops -triple amdgcn-amd-amdhsa -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s
% REQUIRES: amdgpu-registered-target

:- module omp_target_compare_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int.

:- pred less(T::in, T::in) is semidet.
less(X, Y) :- X @< Y.
% CHECK: error: 'merk.compare_word' op compares values of a type variable, which cannot be done on an OpenMP device yet

:- pred fill_it(int::in, array(int)::array_di, array(int)::array_uo) is det.
fill_it(N, !A) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        ( if less(I, 1) then V = 1 else V = 0 ),
        !A ^ unsafe_elem(I) := V
    ).

main(!IO) :-
    array.init(2, 0, A0),
    fill_it(2, A0, A),
    io.write_int(array.lookup(A, 1), !IO),
    io.nl(!IO).
