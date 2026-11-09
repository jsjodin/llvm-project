% A value of a type of constants leaves a target region as one word, whichever
% branch of the region chose it. The output is mmc's.
%
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.optimized && %libomp-env %t.optimized | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: libomp

:- module omp_target_enum_output.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- pred choose(int::in, colour::out, int::out, io::di, io::uo) is det.
choose(N, Colour, Double, !IO) :-
    %$omp target
    (
        Double = N * 2,
        ( if N < 1 then
            Colour = red
        else if N < 2 then
            Colour = green
        else
            Colour = blue
        )
    ).

:- pred show(int::in, io::di, io::uo) is det.
show(N, !IO) :-
    choose(N, Colour, Double, !IO),
    io.write(Colour, !IO),
    io.write_string(" ", !IO),
    io.write_int(Double, !IO),
    io.nl(!IO).

main(!IO) :-
    show(0, !IO),
    show(1, !IO),
    show(2, !IO).

% OUTPUT:red 0
% OUTPUT-NEXT:green 2
% OUTPUT-NEXT:blue 4
% OUTPUT-NOT:{{.}}
