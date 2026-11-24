% MANUAL: Solver types and negated contexts
% Negations and conditions with nonlocals of inst any, in promise_pure
% scopes, and an any_pred expression called in a condition, which mmc
% lets go without a promise (design document, the register).
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for values of type nyi_promised.v

:- module nyi_promised.
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
p(V) :- promise_pure ( not c(V) ).
:- pred q(v::ia) is semidet.
q(V) :- Q = (any_pred is semidet :- c(V)), ( if Q then true else fail ).

main(!IO) :-
    new(V),
    promise_pure (
        ( if p(V) then io.write_string("p\n", !IO) else io.write_string("not p\n", !IO) ),
        ( if q(V) then io.write_string("q\n", !IO) else io.write_string("not q\n", !IO) )
    ).
