% MANUAL: Constrained polymorphic modes
% A call whose argument's inst is not compatible with the inst parameter's
% constraint is a mode error.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:23:{{[0-9]+}}: error:

:- module reject_violated_constraint.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- pred keep(T, T).
:- mode keep(in(I =< bound(yes)), out(I =< bound(yes))) is det.
keep(X, X).

:- type answer
    --->    yes
    ;       no.

main(!IO) :-
    keep(no, A),                                    % ERROR
    io.write_line(A, !IO).
