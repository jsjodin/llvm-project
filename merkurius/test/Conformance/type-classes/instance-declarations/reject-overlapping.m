% MANUAL: Instance declarations
% A program may not have two instances of a class for one type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:20:{{[0-9]+}}: error:

:- module reject_overlapping.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T) where [
    func f(T) = int
].

:- instance c(int) where [
    f(_) = 0
].

:- instance c(int) where [                          % ERROR
    f(_) = 1
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
