% MANUAL: Abstract instance declarations
% An abstract instance declaration must be accompanied in the
% implementation by the instance that defines its methods.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_no_definition.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- typeclass c(T) where [
    func f(T) = int
].
:- instance c(int).                                 % ERROR
:- implementation.
main(!IO) :-
    io.write_string("hello\n", !IO).
