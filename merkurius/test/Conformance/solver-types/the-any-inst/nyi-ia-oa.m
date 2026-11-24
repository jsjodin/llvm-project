% MANUAL: The any inst
% Solver values made with inst any by a foreign procedure's oa argument and
% passed on as ia, in a promise_pure scope where they are in a condition.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for values of type nyi_ia_oa.v

:- module nyi_ia_oa.
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

:- pred same(v::ia, v::oa) is det.
same(V, V).

main(!IO) :-
    new(V0), same(V0, V),
    promise_pure (
        if c(V) then io.write_string("zero\n", !IO)
        else io.write_string("not zero\n", !IO)
    ).
