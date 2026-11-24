% MANUAL: Type class constraints on predicates and functions
% A variable of a constraint must appear in the signature.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_variable_not_in_signature.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T) where [
    func f(T) = int
].

:- pred p(int::out) is det <= c(T).                 % ERROR
p(0).

main(!IO) :-
    p(X),
    io.write_int(X, !IO).
