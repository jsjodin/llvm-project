% MANUAL: The any inst
% A term built from a value of inst any has inst any too, so it may not be
% passed as in.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:32:{{[0-9]+}}: error:

:- module reject_term_of_any.
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

:- pred h({v}::in) is semidet.
h(_) :- semidet_true.

main(!IO) :-
    new(V), T = {V},
    promise_pure (
        if h(T) then io.write_string("a\n", !IO) else true % ERROR
    ).
