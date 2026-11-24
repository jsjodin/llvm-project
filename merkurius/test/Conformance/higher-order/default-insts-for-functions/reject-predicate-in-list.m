% MANUAL: Default insts for functions
% There is no default inst for predicates: a predicate term taken out of a
% polymorphic collection cannot be called.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:18:{{[0-9]+}}: error:

:- module reject_predicate_in_list.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    Ps = [(pred(X::out) is det :- X = 1)],
    P = list.det_head(Ps),
    call(P, Y),                                     % ERROR
    io.write_line(Y, !IO).
