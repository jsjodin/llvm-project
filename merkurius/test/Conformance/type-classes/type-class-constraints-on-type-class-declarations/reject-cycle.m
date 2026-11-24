% MANUAL: Type class constraints on type class declarations
% The superclass relation must be acyclic.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_cycle.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass a(T) <= b(T) where [                   % ERROR
    func f(T) = int
].

:- typeclass b(T) <= a(T) where [
    func g(T) = int
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
