% MANUAL: Module-local mutable variables
% A constant mutable has only a pure get predicate.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: desugaring mutable declarations

:- module nyi_constant.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- mutable(primes, list(int), [2, 3, 5], ground, [constant]).

main(!IO) :-
    get_primes(Ps),
    io.write_line(Ps, !IO).
