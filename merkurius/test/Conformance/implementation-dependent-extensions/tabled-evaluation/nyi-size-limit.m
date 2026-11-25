% MANUAL: Tabled evaluation
% mmc's size_limit attribute, which the manual does not describe, with a
% negative limit, which mmc accepts.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: tabled evaluation (`:- pragma memo')

:- module nyi_size_limit.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- func double(int) = int.
:- pragma memo(double/1, [size_limit(-1)]).
double(N) = 2 * N.

main(!IO) :- io.write_int(double(21), !IO), io.nl(!IO).
