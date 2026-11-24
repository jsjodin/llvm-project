% MANUAL: Type class constraints on predicates and functions
% A call of a constrained predicate at a type with no instance is an
% error.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:21:{{[0-9]+}}: error:

:- module reject_unsatisfied.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T) where [
    func f(T) = int
].

:- func g(T) = int <= c(T).
g(X) = f(X).

main(!IO) :-
    io.write_int(g(1.5), !IO).                      % ERROR
