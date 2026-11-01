% A unique value put in a term while both stay used: the two are one
% value, so that a destructive update of the value would change the term,
% and mmc's mode analysis, which keeps the value unique (-fmmc-uniqueness),
% compiles code that does. By default, the value is shared, as a copy of
% it made by an assignment is in mmc too; the term's copy is shared either
% way. A value that is not used after, or a term that is not, stays
% unique, and a field taken out of a term used after is shared, as in mmc.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: %s < %t.err
% RUN: not merkc -fc1 -fdebug-dump-determinism -fmmc-uniqueness %s 2>%t.mmc >/dev/null
% RUN: FileCheck --check-prefix=MMC --implicit-check-not=error: %s < %t.mmc
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module aliased.
:- interface.
:- type t ---> t(int).
:- type p ---> p(t, t).
:- inst up == unique(p(unique, unique)).
:- pred clobbered_in_term(t::di, t::uo) is det.
:- pred term_used_first(t::di, t::uo) is det.
:- pred output_in_term(t::uo, p::out) is det.
:- pred moved(t::di, p::out(up)) is det.
:- pred term_unused(t::di, t::uo) is det.
:- pred assigned(t::di, t::uo) is det.
:- pred taken_out(t::di, t::uo) is det.
:- pred taken_out_unused(t::di, t::uo) is det.
:- implementation.
:- import_module int.
:- pred updu(int::in, t::di, t::uo) is det.
updu(Y, t(X), t(X + Y)).
:- pred g(t::in) is det.
g(_).
:- pred gp(p::in) is det.
gp(_).

clobbered_in_term(T0, T) :- P = p(T0, t(2)), updu(1, T0, T), gp(P).
% ERR-DAG: aliased.m:[[@LINE-1]]:46: error: in the clause for clobbered_in_term(di, uo): mode error in argument 2 of the call to pred aliased.updu/3: T0 has inst ground, but unique is expected
term_used_first(T0, T) :- P = p(T0, t(2)), gp(P), updu(1, T0, T).
% ERR-DAG: aliased.m:[[@LINE-1]]:51: error: in the clause for term_used_first(di, uo): mode error in argument 2 of the call to pred aliased.updu/3: T0 has inst ground, but unique is expected
output_in_term(T, P) :- T = t(1), P = p(T, t(2)).
% ERR-DAG: aliased.m:[[@LINE-1]]:39: error: in the clause for output_in_term(uo, out): mode error: argument 1, T, had the wrong inst: its final inst is bound(t(bound(1))), but unique is expected
moved(T0, P) :- P = p(T0, t(2)).
term_unused(T0, T) :- _ = p(T0, t(2)), updu(1, T0, T).
assigned(T0, T) :- T1 = T0, g(T1), updu(1, T0, T).
% ERR-DAG: aliased.m:[[@LINE-1]]:36: error: in the clause for assigned(di, uo): mode error in argument 2 of the call to pred aliased.updu/3: T0 has inst ground, but unique is expected
% MMC-DAG: aliased.m:[[@LINE-2]]:36: error: in the clause for assigned(di, uo): mode error in argument 2 of the call to pred aliased.updu/3: T0 has inst ground, but unique is expected
taken_out(T0, T) :- P0 = p(T0, t(1)), P0 = p(A0, _), updu(1, A0, T), gp(P0).
% ERR-DAG: aliased.m:[[@LINE-1]]:54: error: in the clause for taken_out(di, uo): mode error in argument 2 of the call to pred aliased.updu/3: A0 has inst ground, but unique is expected
% MMC-DAG: aliased.m:[[@LINE-2]]:54: error: in the clause for taken_out(di, uo): mode error in argument 2 of the call to pred aliased.updu/3: A0 has inst ground, but unique is expected
taken_out_unused(T0, T) :- P0 = p(T0, t(1)), P0 = p(A0, _), updu(1, A0, T).
