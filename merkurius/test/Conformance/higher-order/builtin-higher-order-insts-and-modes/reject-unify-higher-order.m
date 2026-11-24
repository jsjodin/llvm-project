% MANUAL: Builtin higher-order insts and modes
% Unifying two higher-order terms is an error, as in the manual's
% foo((pred(X::out) is det :- X = 6)).
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:25:{{[0-9]+}}: error:

:- module reject_unify_higher_order.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pred sum(list(int)::in, int::out) is det.
sum(List, Sum) :-
    list.foldl((pred(E::in, A0::in, A::out) is det :- A = A0 + E), List, 0,
        Sum).

:- pred foo(pred(int)).
:- mode foo(free >> (pred(out) is det)) is det.
foo(sum([1, 2, 3])).

main(!IO) :-
    foo((pred(X::out) is det :- X = 6)),            % ERROR
    io.write_string("unreachable\n", !IO).
