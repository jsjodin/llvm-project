% A module that declares predicates and functions it does not define, for
% undefined-calls.m, beside ones it defines by a clause, with a state
% variable, by a DCG rule, by a function clause and by a C foreign_proc.
:- module declared_only.
:- interface.
:- import_module io, list.
:- pred undefined_pred(int::in, io::di, io::uo) is det.
:- func undefined_func(int) = int.
:- pred defined_by_state(int::in, io::di, io::uo) is det.
:- pred defined_by_dcg(list(int)::in, list(int)::out) is det.
:- func defined_func(int) = int.
:- func defined_foreign(int) = int.
:- implementation.
:- import_module int.

defined_by_state(N, !IO) :- io.write_int(N, !IO), io.nl(!IO).

defined_by_dcg --> [].

defined_func(N) = N + 1.

:- pragma foreign_proc("C", defined_foreign(N::in) = (M::out),
    [will_not_call_mercury, promise_pure, thread_safe], "M = N * 2;").
