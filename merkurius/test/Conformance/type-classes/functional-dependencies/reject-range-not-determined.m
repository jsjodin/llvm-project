% MANUAL: Functional dependencies
% An instance whose range has a variable not determined by its domain
% breaks the dependency: baz(string, list(T)).
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:19:{{[0-9]+}}: error:

:- module reject_range_not_determined.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- typeclass baz(A, B) <= (A -> B) where [
    func name(A) = string
].

:- instance baz(string, list(T)) where [            % ERROR
    name(_) = "string"
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
