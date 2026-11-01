% Unique values that an if-then-else needs on backtracking. By default,
% merkc finds some that mmc's rules (-fmmc-uniqueness) miss: what the
% condition makes from a value the else part needs is needed too, and
% what is used after an if-then-else is needed if backtracking into it
% can run that again. A value the else part needs is needed only while
% the condition runs, and only if the condition clobbers it or what it
% makes from it (see the design document's register).
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: %s < %t.err
% RUN: not merkc -fc1 -fdebug-dump-determinism -fmmc-uniqueness %s 2>%t.mmc >/dev/null
% RUN: FileCheck --check-prefix=MMC --implicit-check-not=error: %s < %t.mmc
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module sound.
:- interface.
:- type t ---> t(int).

:- pred alias_in_condition(t::di, t::muo) is det.
:- pred after_condition(t::di, t::uo) is multi.
:- pred after_then(t::di, t::uo) is multi.
:- pred after_else(t::di, t::uo) is multi.
:- pred det_parts(t::di, t::uo) is det.
:- pred alias_in_then(t::di, t::muo) is det.
:- pred else_value_in_then(t::di, t::uo) is det.
:- pred retried_condition(t::di, t::uo) is nondet.

:- implementation.
:- import_module int.

:- pred updu(int::in, t::di, t::uo) is det.
updu(Y, t(X), t(X + Y)).
:- pred peek(t::ui, int::out) is det.
peek(_, 1).
:- pred m(int::out) is multi.
m(1).
m(2).

% ERR-DAG: sound.m:[[@LINE+1]]:44: error: in the clause for alias_in_condition(di, muo): mode error in argument 2 of the call to pred sound.updu/3: T4 has inst mostly_unique, but unique is expected
alias_in_condition(T0, T) :- ( if T4 = T0, updu(1, T4, T5), 1 > 2 then T = T5 else T = T0 ).
% ERR-DAG: sound.m:[[@LINE+1]]:69: error: in the clause for after_condition(di, uo): mode error in argument 2 of the call to pred sound.updu/3: T0 has inst mostly_unique, but unique is expected
after_condition(T0, T) :- ( if m(X), X > 1 then Y = X else Y = 0 ), updu(Y, T0, T).
% ERR-DAG: sound.m:[[@LINE+1]]:57: error: in the clause for after_then(di, uo): mode error in argument 2 of the call to pred sound.updu/3: T0 has inst mostly_unique, but unique is expected
after_then(T0, T) :- ( if 1 > 2 then m(Y) else Y = 2 ), updu(Y, T0, T).
% ERR-DAG: sound.m:[[@LINE+1]]:57: error: in the clause for after_else(di, uo): mode error in argument 2 of the call to pred sound.updu/3: T0 has inst mostly_unique, but unique is expected
after_else(T0, T) :- ( if 1 > 2 then Y = 2 else m(Y) ), updu(Y, T0, T).
% Accepted: nothing is retried, or the then part uses what only the
% else part needs.
det_parts(T0, T) :- ( if 1 > 2 then Y = 2 else Y = 3 ), updu(Y, T0, T).
alias_in_then(T0, T) :- ( if T4 = T0, 1 > 2 then updu(1, T4, T) else T = T0 ).
else_value_in_then(T0, T) :- ( if peek(T0, X), X > 1 then updu(1, T0, T) else T = T0 ).
% Rejected by both: the condition is retried after the then part.
% ERR-DAG: sound.m:[[@LINE+2]]:40: error: in the clause for retried_condition(di, uo): mode error in argument 1 of the call to pred sound.peek/2: T0 has inst mostly_unique, but unique is expected
% MMC-DAG: sound.m:[[@LINE+1]]:40: error: in the clause for retried_condition(di, uo): mode error in argument 1 of the call to pred sound.peek/2: T0 has inst mostly_unique, but unique is expected
retried_condition(T0, T) :- ( if m(X), peek(T0, Y) then updu(Y, T0, T), X > 1 else T = T0 ).
