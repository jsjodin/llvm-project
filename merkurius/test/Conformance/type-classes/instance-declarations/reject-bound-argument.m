% MANUAL: Instance declarations
% An instance's type must have only type variables as arguments: not
% list(int).
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:19:{{[0-9]+}}: error:

:- module reject_bound_argument.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- typeclass c(T) where [
    func f(T) = int
].

:- instance c(list(int)) where [                    % ERROR
    f(_) = 0
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
