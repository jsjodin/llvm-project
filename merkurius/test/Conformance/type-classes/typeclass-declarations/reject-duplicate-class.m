% MANUAL: Typeclass declarations
% A module may not declare two type classes of the same name and
% arity.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_duplicate_class.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T) where [
    func f(T) = T
].

:- typeclass c(T) where [                           % ERROR
    func g(T) = T
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
