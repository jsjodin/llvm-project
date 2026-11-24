% MANUAL: Type class constraints on type class declarations
% A variable of a superclass constraint must be a parameter of the
% class.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_variable_not_parameter.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass base(T) where [
    func f(T) = int
].

:- typeclass derived(T) <= base(U) where [          % ERROR
    func g(T) = int
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
