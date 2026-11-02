% A closure passed where a higher-order inst is expected, as in mmc: its
% modes must accept the expected ones, so that each argument's expected
% initial inst matches the closure's, and the closure's final inst
% matches the expected one. So a named inst that expands to the expected
% one is accepted, as `di(ut)` with `ut == unique` for `di`, and a closure
% that asks for less, `ground >> clobbered` for `di`; one that gives less,
% `out` for `uo`, or clobbers what is expected to stay ground, is not, nor
% is one that asks for more, `di` for `in`. A call with several modes
% takes the first whose closure inst matches.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>%t.err >/dev/null
% RUN: FileCheck --implicit-check-not=error: %s < %t.err
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module closure_variance.
:- interface.
:- type t ---> t(int).
:- inst ut == unique.
:- pred same(t::di, t::uo) is det.
:- pred named(t::di, t::uo) is det.
:- pred spelled_out(t::di, t::uo) is det.
:- pred asks_less(t::di, t::uo) is det.
:- pred asks_less_named(t::di, t::uo) is det.
:- pred named_initial(t::di, t::uo) is det.
:- pred gives_less(t::di, t::uo) is det.
:- pred asks_more(t::in, t::out) is det.
:- pred clobbers_ground(t::in, t::out) is det.
:- pred mostly_clobbers_ground(t::in, t::out) is det.
:- pred clobbers_mostly(t::in, t::out) is det.
:- pred gives_more(t::in, t::out) is det.
:- pred no_mode_fits(t::di, t::uo) is det.
:- pred no_mode_fits_named(t::di, t::uo) is det.
:- implementation.
:- pred app(pred(t, t)::in(pred(di, uo) is det), t::di, t::uo) is det.
app(P, T0, T) :- P(T0, T).
:- pred app_in(pred(t, t)::in(pred(in, out) is det), t::in, t::out) is det.
app_in(P, T0, T) :- P(T0, T).
:- pred app_mostly(pred(t, t)::in(pred((ground >> mostly_clobbered), out) is det),
    t::in, t::out) is det.
app_mostly(P, T0, T) :- P(T0, T).
% CHECK-DAG: closure-variance.m:[[@LINE-1]]:25: error: in the clause for app_mostly(in((pred((ground >> mostly_clobbered), out) is det)), in, out): unique-mode error in argument 2 of the higher-order predicate call: the called procedure would clobber its argument, but T0 is still live
:- pred app_clobbered(pred(t, t)::in(pred(in, (free >> clobbered)) is det),
    t::in, t::out) is det.
app_clobbered(P, T0, T) :- P(T0, _), T = T0.
:- pred two(pred(t, t), t, t).
:- mode two(in(pred(in, out) is det), in, out) is det.
:- mode two(in(pred(di, uo) is det), di, uo) is det.
two(P, T0, T) :- P(T0, T).

same(T0, T) :- app(pred(A::di, B::uo) is det :- B = A, T0, T).
named(T0, T) :- app(pred(A::di(ut), B::out(ut)) is det :- B = A, T0, T).
spelled_out(T0, T) :- app(pred(A::(unique >> clobbered), B::(free >> unique)) is det :- B = A, T0, T).
asks_less(T0, T) :- app(pred(_::(ground >> clobbered), B::(free >> unique)) is det :- B = t(1), T0, T).
asks_less_named(T0, T) :- app(pred(_::(ground >> clobbered), B::out(ut)) is det :- B = t(1), T0, T).
named_initial(T0, T) :- app(pred(_::di(ut), B::(free >> unique)) is det :- B = t(1), T0, T).
gives_less(T0, T) :- app(pred(A::in, B::out) is det :- B = A, T0, T).
% CHECK-DAG: closure-variance.m:[[@LINE-1]]:22: error: in the clause for gives_less(di, uo): mode error in argument 1 of the call to pred closure_variance.app/3: V_2 has inst /* unique */ (pred(in, out) is det), but (pred(di, uo) is det) is expected
asks_more(T0, T) :- app_in(pred(A::di, B::uo) is det :- B = A, T0, T).
% CHECK-DAG: closure-variance.m:[[@LINE-1]]:21: error: in the clause for asks_more(in, out): mode error in argument 1 of the call to pred closure_variance.app_in/3: V_2 has inst /* unique */ (pred(di, uo) is det), but (pred(in, out) is det) is expected
clobbers_ground(T0, T) :- app_in(pred(_::(ground >> clobbered), B::out) is det :- B = t(1), T0, T).
% CHECK-DAG: closure-variance.m:[[@LINE-1]]:27: error: in the clause for clobbers_ground(in, out): mode error in argument 1 of the call to pred closure_variance.app_in/3: V_2 has inst /* unique */ (pred(di(ground), out) is det), but (pred(in, out) is det) is expected
mostly_clobbers_ground(T0, T) :- app_in(pred(_::(ground >> mostly_clobbered), B::out) is det :- B = t(1), T0, T).
% CHECK-DAG: closure-variance.m:[[@LINE-1]]:34: error: in the clause for mostly_clobbers_ground(in, out): mode error in argument 1 of the call to pred closure_variance.app_in/3: V_2 has inst /* unique */ (pred((ground >> mostly_clobbered), out) is det), but (pred(in, out) is det) is expected
clobbers_mostly(T0, T) :- app_mostly(pred(_::(ground >> clobbered), B::out) is det :- B = t(1), T0, T).
% CHECK-DAG: closure-variance.m:[[@LINE-1]]:27: error: in the clause for clobbers_mostly(in, out): mode error in argument 1 of the call to pred closure_variance.app_mostly/3: V_2 has inst /* unique */ (pred(di(ground), out) is det), but (pred((ground >> mostly_clobbered), out) is det) is expected
gives_more(T0, T) :- app_clobbered(pred(A::in, B::(free >> ground)) is det :- B = A, T0, T).
no_mode_fits(T0, T) :- two(pred(B0::(ground >> clobbered), B::(free >> ground)) is det :- B = B0, T0, T).
% CHECK-DAG: closure-variance.m:[[@LINE-1]]:24: error: in the clause for no_mode_fits(di, uo): mode error in the call to pred closure_variance.two/3: the arguments V_2, T0, T have insts /* unique */ (pred(di(ground), out) is det), unique, free, which match none of its modes
no_mode_fits_named(T0, T) :- P = (pred(B0::di, B::out) is det :- B = B0), two(P, T0, T).
% CHECK-DAG: closure-variance.m:[[@LINE-1]]:75: error: in the clause for no_mode_fits_named(di, uo): mode error in the call to pred closure_variance.two/3: the arguments P, T0, T have insts /* unique */ (pred(di, out) is det), unique, free, which match none of its modes
