% A goal that uses a variable wakes the goals that would clobber it and
% wait for it to be used no more, even when its inst stays the same, as a
% ground one's does, as in mmc: an assignment from it, or a call that
% reads it; a goal waiting for its inst to change, as a call whose modes
% do not match yet, is not woken. So the clobbering call runs right after
% that goal, and the rest of the conjunction after it, so that a final
% inst error is reported at the clause's last goal, as in mmc. q6's final
% inst is an error only with -fmmc-uniqueness, as a term with a number in
% it is unique by default.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>%t.err >/dev/null
% RUN: FileCheck --implicit-check-not=error: %s < %t.err
% RUN: not merkc -fc1 -fdebug-dump-modes -fmmc-uniqueness %s 2>%t.mmc.err >/dev/null
% RUN: FileCheck --check-prefixes=CHECK,MMC --implicit-check-not=error: %s < %t.mmc.err
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module assignment_wakes.
:- interface.
:- type t ---> t(int).
:- pred q1(t::(ground >> mostly_clobbered), t::out) is det.
:- pred q2(t::(ground >> mostly_clobbered), t::out) is det.
:- pred q3(t::(unique >> mostly_clobbered), t::out) is det.
:- pred q4(t::(ground >> mostly_clobbered), t::out) is det.
:- pred q5(t::(ground >> mostly_clobbered), t::out) is det.
:- pred q6(t::di, t::uo) is nondet.
:- implementation.
:- import_module int.
:- pred updg(int::in, t::(ground >> clobbered), t::out) is det.
updg(Y, t(X), t(X + Y)).
:- pred peek(t::in, int::out) is det.
peek(t(X), X).
q1(T0, T) :-
    updg(1, T0, T3),
    peek(T3, _),
    T6 = T0,
    updg(2, T6, T7),
    T = T7.
% CHECK-DAG: assignment-wakes.m:[[@LINE-1]]:5: error: in the clause for q1((ground >> mostly_clobbered), out): mode error: argument 1, T0, did not get sufficiently instantiated: its final inst is clobbered, but mostly_clobbered is expected
q2(T0, T) :-
    updg(1, T0, T3),
    peek(T3, _),
    T6 = T0,
    T = T6.
% CHECK-DAG: assignment-wakes.m:[[@LINE-1]]:5: error: in the clause for q2((ground >> mostly_clobbered), out): mode error: argument 1, T0, did not get sufficiently instantiated: its final inst is clobbered, but mostly_clobbered is expected
q3(T0, T) :-
    updg(1, T0, T3),
    peek(T3, _),
    T6 = T0,
    updg(2, T6, T7),
    T = T7.
% CHECK-DAG: assignment-wakes.m:[[@LINE-1]]:5: error: in the clause for q3((unique >> mostly_clobbered), out): mode error: argument 1, T0, did not get sufficiently instantiated: its final inst is clobbered, but mostly_clobbered is expected
q4(T0, T) :-
    updg(1, T0, T3),
    peek(T3, _),
    peek(T0, Y),
    T5 = t(Y),
    T = T5.
% CHECK-DAG: assignment-wakes.m:[[@LINE-1]]:5: error: in the clause for q4((ground >> mostly_clobbered), out): mode error: argument 1, T0, did not get sufficiently instantiated: its final inst is clobbered, but mostly_clobbered is expected
q5(T0, T) :-
    updg(1, T0, T3),
    peek(T0, Y),
    peek(T3, _),
    T5 = t(Y),
    T = T5.
% CHECK-DAG: assignment-wakes.m:[[@LINE-1]]:5: error: in the clause for q5((ground >> mostly_clobbered), out): mode error: argument 1, T0, did not get sufficiently instantiated: its final inst is clobbered, but mostly_clobbered is expected
:- pred m(int::out) is multi.
m(1).
m(2).
:- pred updu(int::in, t::di, t::uo) is det.
updu(_, T, T).
:- pred keep(t::di, t::uo) is det.
keep(T, T).
:- pred choose(t::di, t::uo) is nondet.
choose(T, T).
q6(T0, T) :-
    keep(T0, T1),
    choose(T1, T2),
    T3 = T2,
    m(X),
    \+ ( m(_), updu(1, T1, _) ),
    T4 = t(X),
    T = T4.
% MMC-DAG: assignment-wakes.m:[[@LINE-1]]:5: error: in the clause for q6(di, uo): mode error: argument 2, T, did not get sufficiently instantiated: its final inst is unique(t(ground)), but unique is expected
