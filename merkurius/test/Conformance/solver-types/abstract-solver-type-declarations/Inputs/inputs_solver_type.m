:- module inputs_solver_type.
:- interface.
:- solver type v.
:- pred new(v::oa) is det.
:- pred zero(v::ia) is semidet.
:- implementation.
:- solver type v where representation is int.
:- pragma foreign_proc("C", new(V::oa),
    [promise_pure, will_not_call_mercury], "V = 0;").
:- pragma foreign_proc("C", zero(V::ia),
    [promise_pure, will_not_call_mercury], "SUCCESS_INDICATOR = V == 0;").
