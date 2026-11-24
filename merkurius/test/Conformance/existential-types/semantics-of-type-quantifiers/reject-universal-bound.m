% MANUAL: Semantics of type quantifiers
% A universally typed variable is opaque in its predicate's body: it may
% not be used as an int.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_universal_bound.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- pred inc(T::in, int::out) is det.
inc(X, Y) :-
    Y = X + 1.                                      % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
