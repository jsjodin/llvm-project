% MANUAL: Instance declarations
% A method called at a type with no instance of its class is an
% error.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:22:{{[0-9]+}}: error:

:- module reject_no_instance.
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

main(!IO) :-
    io.write_int(f("text"), !IO).                   % ERROR
