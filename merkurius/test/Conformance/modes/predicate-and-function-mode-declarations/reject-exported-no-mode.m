% MANUAL: Predicate and function mode declarations
% A predicate exported from a module must have a mode declaration, in the
% interface, as the mode declarations of a predicate declared there must be.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_exported_no_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- pred p(int, int).                                % ERROR
:- implementation.

p(X, X).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
