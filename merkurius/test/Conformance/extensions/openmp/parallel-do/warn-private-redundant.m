% DESIGN: Loops / Under a parallel directive
% A variable that occurs only in a parallel loop's body is local to each
% iteration, what OpenMP calls private, with nothing to declare, and so is
% the for's counter. A private clause naming one is redundant and draws a
% warning; the loop is generated as without it.
% ORACLE: none
%
% REQUIRES: libomp
% RUN: merkc -O2 -fdo-loops -fopenmp %s %libomp-flags -o %t 2>&1 | FileCheck %s --implicit-check-not=warning: --implicit-check-not=error:
% RUN: %libomp-env OMP_NUM_THREADS=4 %t 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
:- module warn_private_redundant.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, list, string.

main(!IO) :-
    N = 1000,
    Step = 1.0 / float(N),
    %$omp parallel do private(X, Y) reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        X = (float(I) + 0.5) * Step,
        Y = 4.0 / (1.0 + X * X),
        S1 = S0 + Y
    ),
% CHECK: warn-private-redundant.m:[[@LINE-6]]:{{[0-9]+}}: warning: X is local to each iteration of the loop already, so naming it in a private clause is redundant
% CHECK: warn-private-redundant.m:[[@LINE-7]]:{{[0-9]+}}: warning: Y is local to each iteration of the loop already, so naming it in a private clause is redundant
    Pi = Step * Sum,
    io.format("%.6f\n", [f(Pi)], !IO),
    %$omp parallel do private(J) reduction((+) : T)
    ( for(J, 1, N), fromto(0, T0, T1, T) do T1 = T0 + J ),
% CHECK: warn-private-redundant.m:[[@LINE-2]]:{{[0-9]+}}: warning: J is local to each iteration of the loop already, so naming it in a private clause is redundant
    io.write_int(T, !IO), io.nl(!IO).
% OUTPUT:3.141593
% OUTPUT-NEXT:500500
% OUTPUT-NOT:{{.}}
