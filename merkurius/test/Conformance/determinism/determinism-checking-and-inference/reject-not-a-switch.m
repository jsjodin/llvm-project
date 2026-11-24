% MANUAL: Determinism checking and inference
% A disjunction whose disjunct calls a predicate before the test of the
% variable is not a switch, so it may succeed more than once.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:21:{{[0-9]+}}: error:

:- module reject_not_a_switch.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pred empty(int::out) is det.
empty(0).

:- pred nonempty(int::in, list(int)::in, int::out) is det.
nonempty(H, _, H).

:- pred not_switch(list(int)::in, int::out) is det. % ERROR
not_switch(L, Out) :-
    (
        empty(Out),
        L = []
    ;
        L = [H | T],
        nonempty(H, T, Out)
    ).

main(!IO) :-
    not_switch([], Out),
    io.write_line(Out, !IO).
