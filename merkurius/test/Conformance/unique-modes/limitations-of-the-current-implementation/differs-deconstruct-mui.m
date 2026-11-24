% MANUAL: Limitations of the current implementation
% DIFFERS: implementation: mmc cannot deconstruct a mui argument, as it
% cannot a ui one: its fields become ground, so the argument no longer
% matches mostly_unique at the end. merkc keeps it mostly unique. Reading
% a mostly unique value's field through mui in a disjunction.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[11, 20]
% OUTPUT-NOT:{{.}}
% MMC-STATUS: rejected
% MMC: 31: In clause for `value(mui, out)': mode error: argument 1 did not get sufficiently instantiated. Final instantiatedness of `HeadVar__1' was `mostly_unique(counter(ground))', expected final instantiatedness was `mostly_unique'.

:- module differs_deconstruct_mui.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions.

:- type counter
    --->    counter(int).

:- pred make(counter::muo) is det.
make(counter(10)).

:- pred value(counter::mui, int::out) is det.
value(counter(N), N).

:- pred add(int::in, counter::mdi, counter::muo) is det.
add(I, counter(N), counter(N + I)).

:- pred choice(int::out) is multi.
choice(V) :-
    make(C0),
    value(C0, N),
    ( I = 1 ; I = N ),
    add(I, C0, C),
    value(C, V).

main(!IO) :-
    solutions(choice, Vs),
    io.write_line(Vs, !IO).
