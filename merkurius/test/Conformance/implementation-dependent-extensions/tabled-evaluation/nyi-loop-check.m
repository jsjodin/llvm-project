% MANUAL: Tabled evaluation
% A loop_check pragma on a predicate whose calls do not loop.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: tabled evaluation (`:- pragma loop_check')

:- module nyi_loop_check.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred count(int::in, int::out) is det.
:- pragma loop_check(count/2).
count(N, M) :- ( if N > 5 then M = N else count(N + 1, M) ).

main(!IO) :- count(0, M), io.write_int(M, !IO), io.nl(!IO).
