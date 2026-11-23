% MANUAL: Constrained polymorphic modes
% An occurrence of an inst parameter without a constraint, in a declaration
% with no `<=` constraint for it, is constrained to ground, so it differs
% from an occurrence constrained otherwise.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:20:{{[0-9]+}}: error:

:- module reject_bare_occurrence.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- type answer
    --->    yes
    ;       no.

:- pred keep(answer, answer).
:- mode keep(in(I =< bound(yes)), out(I)) is det.   % ERROR
keep(X, X).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
