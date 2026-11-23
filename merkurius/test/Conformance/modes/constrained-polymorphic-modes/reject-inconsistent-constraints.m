% MANUAL: Constrained polymorphic modes
% An inst parameter that occurs more than once in a declaration must have
% the same constraint at each occurrence.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:19:{{[0-9]+}}: error:

:- module reject_inconsistent_constraints.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- type answer
    --->    yes
    ;       no.

:- pred keep(answer, answer).
:- mode keep(in(I =< ground), out(I =< bound(yes))) is det.     % ERROR
keep(X, X).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
