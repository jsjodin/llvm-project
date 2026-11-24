% MANUAL: Instance declarations
% An instance may not define a method twice: by naming a function and by
% clauses.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:20:{{[0-9]+}}: error:

:- module reject_method_twice.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T) where [
    func f(T) = int
].

:- func zero(int) = int.
zero(_) = 0.

:- instance c(int) where [                          % ERROR
    func(f/1) is zero,
    f(_) = 1
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
