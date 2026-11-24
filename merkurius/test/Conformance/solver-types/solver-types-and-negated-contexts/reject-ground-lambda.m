% MANUAL: Solver types and negated contexts
% A pred expression is ground, so its nonlocals may not have inst any.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:26:{{[0-9]+}}: error:

:- module reject_ground_lambda.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- solver type v where representation is int.
:- pred new(v::oa) is det.
:- pragma foreign_proc("C", new(V::oa),
    [promise_pure, will_not_call_mercury], "V = 0;").
:- pred c(v::ia) is semidet.
:- pragma foreign_proc("C", c(V::ia),
    [promise_pure, will_not_call_mercury], "SUCCESS_INDICATOR = V == 0;").
:- pred g(v::in) is semidet.
:- pragma foreign_proc("C", g(V::in),
    [promise_pure, will_not_call_mercury], "SUCCESS_INDICATOR = V == 0;").

:- pred p(v::ia) is semidet.
p(V) :- Q = ((pred) is semidet :- c(V)), Q. % ERROR

main(!IO) :-
    new(V),
    promise_pure ( if p(V) then io.write_string("a\n", !IO) else true ).
