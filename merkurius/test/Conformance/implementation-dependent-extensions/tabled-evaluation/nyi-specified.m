% MANUAL: Tabled evaluation
% A memo pragma for one mode, with an argument tabled by value, one not
% tabled, and the output.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: tabled evaluation (`:- pragma memo')

:- module nyi_specified.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred add(int, int, int).
:- mode add(in, in, out) is det.
:- mode add(out, in, in) is det.
:- pragma memo(add(in, in, out), [specified([value, promise_implied, output])]).
:- pragma promise_equivalent_clauses(add/3).
add(X::in, Y::in, (X + Y)::out).
add((Z - Y)::out, Y::in, Z::in).

main(!IO) :- add(1, 2, Z), add(X, 2, 5), io.write_int(Z + X, !IO), io.nl(!IO).
