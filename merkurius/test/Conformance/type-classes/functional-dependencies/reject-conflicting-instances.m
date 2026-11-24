% MANUAL: Functional dependencies
% Two instances whose domains are the same but whose ranges differ break
% the dependency.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:23:{{[0-9]+}}: error:

:- module reject_conflicting_instances.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- typeclass baz(A, B) <= (A -> B) where [
    func name(A) = string
].

:- instance baz(int, int) where [
    name(_) = "int int"
].

:- instance baz(int, string) where [                % ERROR
    name(_) = "int string"
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
