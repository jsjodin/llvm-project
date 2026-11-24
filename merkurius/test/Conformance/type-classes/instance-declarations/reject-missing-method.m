% MANUAL: Instance declarations
% An instance must define every method of its class.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_missing_method.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T) where [
    func f(T) = int,
    func g(T) = int
].

:- instance c(int) where [                          % ERROR
    f(_) = 0
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
