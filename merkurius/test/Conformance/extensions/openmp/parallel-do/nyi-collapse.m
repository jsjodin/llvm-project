% DESIGN: Loops / Under a parallel directive
% With collapse(N), the body of each of the outer N - 1 loops is exactly
% the next loop, and the nest is one loop over all their iterations: the
% design document's matrix multiplication. Planned, not yet built.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/collapse_translation.m
%
% RUN: not merkc -fdo-loops -fopenmp %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for the collapse clause of the do directive
:- module nyi_collapse.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, string.

:- pred matmul(int::in, array(float)::in, array(float)::in,
    array(float)::array_di, array(float)::array_uo) is det.
matmul(N, A, B, !C) :-
    %$omp parallel do collapse(2)
    ( for(I, 0, N - 1) do
        ( for(J, 0, N - 1) do
            ( for(K, 0, N - 1), fromto(0.0, S0, S1, Sum) do
                S1 = S0 + A ^ unsafe_elem(I * N + K) * B ^ unsafe_elem(K * N + J)
            ),
            !C ^ unsafe_elem(I * N + J) := Sum
        )
    ).

main(!IO) :-
    N = 3,
    A = array.generate(N * N, (func(X) = float(X))),
    B = array.generate(N * N, (func(X) = (if X mod (N + 1) = 0 then 1.0 else 0.0))),
    array.init(N * N, 0.0, C0),
    matmul(N, A, B, C0, C),
    io.write_string(string.join_list(" ",
        list.map(float_to_string, array.to_list(C))), !IO),
    io.nl(!IO).
