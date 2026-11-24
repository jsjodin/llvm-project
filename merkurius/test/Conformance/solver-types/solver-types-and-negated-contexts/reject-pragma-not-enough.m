% MANUAL: Solver types and negated contexts
% A promise_pure pragma does not make a negation in the clause sound; a
% promise scope must be around it.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:28:{{[0-9]+}}: error:

:- module reject_pragma_not_enough.
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
:- pragma promise_pure(p/1).
p(V) :- not c(V). % ERROR

main(!IO) :-
    new(V),
    promise_pure ( if p(V) then io.write_string("a\n", !IO) else true ).
