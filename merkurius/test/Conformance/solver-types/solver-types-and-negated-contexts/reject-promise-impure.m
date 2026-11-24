% MANUAL: Solver types and negated contexts
% A promise_impure scope makes the negation sound, but the predicate impure.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:25:{{[0-9]+}}: error:

:- module reject_promise_impure.
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

:- pred p(v::ia) is semidet. % ERROR
p(V) :- promise_impure ( not c(V) ).

main(!IO) :-
    new(V),
    promise_pure ( if p(V) then io.write_string("a\n", !IO) else true ).
