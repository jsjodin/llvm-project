% MANUAL: Existential class constraints
% A universal constraint may only constrain universally quantified type
% variables.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_universal_on_existential.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T) where [ func name(T) = int ].

:- some [T2] (pred p(T1, T2) <= c(T2)).             % ERROR
:- mode p(in, out) is det.
p(_, 1).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
