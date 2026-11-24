% MANUAL: Instance declarations
% An instance that defines methods may not be in the interface
% section.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_in_interface.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- typeclass c(T) where [
    func f(T) = int
].
:- instance c(int) where [                          % ERROR
    f(_) = 0
].
:- implementation.
main(!IO) :-
    io.write_int(f(1), !IO).
