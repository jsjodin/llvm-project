% MANUAL: Typeclass declarations
% Each predicate method needs a mode declaration.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_no_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T) where [
    pred p(T, int)                                  % ERROR
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
