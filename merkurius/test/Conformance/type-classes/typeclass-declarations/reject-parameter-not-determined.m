% MANUAL: Typeclass declarations
% Each parameter of the class must appear in the type of each method.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_parameter_not_determined.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T) where [
    func size = int                                 % ERROR
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
