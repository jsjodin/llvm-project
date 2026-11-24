% MANUAL: Type class constraints on instance declarations
% The constraints of an abstract instance declaration must be exactly
% those of the instance that defines it.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:20:{{[0-9]+}}: error:

:- module reject_abstract_constraints_differ.
:- interface.
:- import_module io, list.
:- pred main(io::di, io::uo) is det.
:- typeclass c(T) where [
    func f(T) = int
].
:- instance c(list(T)).
:- implementation.
:- instance c(int) where [
    f(X) = X
].
:- instance c(list(T)) <= c(T) where [              % ERROR
    f(_) = 0
].
main(!IO) :-
    io.write_int(f([1]), !IO).
