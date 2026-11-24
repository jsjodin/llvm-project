% MANUAL: Instance declarations
% A function named as a method must be at least as permissive as the
% method: a semidet one cannot implement a det method.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:25:{{[0-9]+}}: error:

:- module reject_weaker_determinism.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- typeclass c(T) where [
    func f(T) = int
].

:- func half(int) = int.
:- mode half(in) = out is semidet.
half(X) = X // 2 :-
    X mod 2 = 0.

:- instance c(int) where [
    func(f/1) is half                               % ERROR
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
