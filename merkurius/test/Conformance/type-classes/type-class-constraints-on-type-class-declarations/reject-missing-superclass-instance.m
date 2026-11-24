% MANUAL: Type class constraints on type class declarations
% An instance of a class needs an instance of each of its
% superclasses.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:21:{{[0-9]+}}: error:

:- module reject_missing_superclass_instance.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass base(T) where [
    func f(T) = int
].

:- typeclass derived(T) <= base(T) where [
    func g(T) = int
].

:- instance derived(int) where [                    % ERROR
    g(_) = 0
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
