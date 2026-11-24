% MANUAL: Backtrackable destructive update
% A mostly dead value may be needed on backtracking, but not in forward
% execution: the first call may not clobber a value the second needs.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:24:{{[0-9]+}}: error:

:- module reject_use_after_mdi.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- import_module int.

:- type counter
    --->    counter(int).

:- pred add(int::in, counter::mdi, counter::muo) is det.
add(I, counter(N), counter(N + I)).

:- pred twice(counter::mdi, counter::muo, counter::muo) is det.
twice(C0, C1, C2) :-
    add(1, C0, C1),                                 % ERROR
    add(2, C0, C2).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
