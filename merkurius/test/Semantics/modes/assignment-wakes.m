% An assignment from a variable wakes the goals that would clobber it and
% wait for it to be used no more, even when its inst stays the same, as a
% ground one's does, as in mmc; a goal waiting for its inst to change, as
% a call whose modes do not match yet, is not woken. So the clobbering
% call runs right after the assignment, and the rest of the conjunction
% after it, so that a final inst error is reported at the clause's last
% goal, as in mmc.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>%t.err >/dev/null
% RUN: FileCheck --implicit-check-not=error: %s < %t.err
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module assignment_wakes.
:- interface.
:- type t ---> t(int).
:- pred q1(t::(ground >> mostly_clobbered), t::out) is det.
:- pred q2(t::(ground >> mostly_clobbered), t::out) is det.
:- pred q3(t::(unique >> mostly_clobbered), t::out) is det.
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
