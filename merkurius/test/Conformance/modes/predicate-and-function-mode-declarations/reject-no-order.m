% MANUAL: Predicate and function mode declarations
% Mode analysis reorders a conjunction to satisfy its calls' modes, and
% reports an error if no order exists.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_no_order.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- pred p(int::out) is det.
p(Y) :-
    Y = X + 1,                                      % ERROR
    X = Y - 1.

main(!IO) :-
    io.write_string("unreachable\n", !IO).
