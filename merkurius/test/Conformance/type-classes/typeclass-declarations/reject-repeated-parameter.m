% MANUAL: Typeclass declarations
% The parameters of a type class must be distinct variables.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_repeated_parameter.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T, T) where [                        % ERROR
    func f(T) = T
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
