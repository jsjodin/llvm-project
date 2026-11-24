% MANUAL: Typeclass declarations
% Each mode declaration of a method must give its determinism.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_no_determinism.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T) where [
    pred p(T, int),
    mode p(in, out)                                 % ERROR
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
