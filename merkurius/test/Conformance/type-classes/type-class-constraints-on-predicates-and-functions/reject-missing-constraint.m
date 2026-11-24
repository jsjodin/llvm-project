% MANUAL: Type class constraints on predicates and functions
% A method called at a type variable needs the constraint in the
% predicate's declaration.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:18:{{[0-9]+}}: error:

:- module reject_missing_constraint.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T) where [
    func f(T) = int
].

:- func g(T) = int.
g(X) = f(X).                                        % ERROR

main(!IO) :-
    io.write_int(g(1), !IO).
