% MANUAL: Tabled evaluation
% fast_loose and specified both choose how arguments are looked up.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_two_lookups.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(int::in, int::in, int::out) is det.
:- pragma memo(p(in, in, out),
    [fast_loose, specified([value, promise_implied, output])]). % ERROR
p(X, Y, X + Y).
main(!IO) :- io.write_string("ok\n", !IO).
