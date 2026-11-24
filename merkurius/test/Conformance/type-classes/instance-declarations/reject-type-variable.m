% MANUAL: Instance declarations
% An instance's type may not be a type variable.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:18:{{[0-9]+}}: error:

:- module reject_type_variable.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- typeclass c(T) where [
    func f(T) = int
].

:- instance c(T) where [                            % ERROR
    f(_) = 0
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
