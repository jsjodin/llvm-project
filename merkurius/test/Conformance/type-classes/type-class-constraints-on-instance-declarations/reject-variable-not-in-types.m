% MANUAL: Type class constraints on instance declarations
% A variable of an instance's constraint must appear in the instance's
% types.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:19:{{[0-9]+}}: error:

:- module reject_variable_not_in_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- typeclass c(T) where [
    func f(T) = int
].

:- instance c(list(T)) <= c(U) where [              % ERROR
    f(_) = 0
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
