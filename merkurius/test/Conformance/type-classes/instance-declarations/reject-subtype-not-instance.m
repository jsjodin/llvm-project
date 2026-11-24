% MANUAL: Instance declarations
% A subtype is not an instance of a class because its supertype is.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:29:{{[0-9]+}}: error:

:- module reject_subtype_not_instance.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T) where [
    func f(T) = int
].

:- type fruit
    --->    apple
    ;       pear.

:- type pome =< fruit
    --->    apple.

:- instance c(fruit) where [
    f(_) = 1
].

main(!IO) :-
    X = apple : pome,
    io.write_int(f(X), !IO).                        % ERROR
