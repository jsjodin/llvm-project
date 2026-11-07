% Field accesses and updates keep uniqueness as the unifications they
% stand for do (reference manual, "Field access functions"), as in mmc: a
% field taken out of a unique term not used after is unique, and so is a
% term updated with a unique value from a unique term not used after. A
% field taken out of a term used after, by a call or by a field access or
% deconstruction of another field, is shared, and so is the update's copy
% of a value used after, as for the deconstructions and constructions
% themselves (aliased.m). mmc reports the same errors.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: %s < %t.err
% RUN: not merkc -fc1 -fdebug-dump-determinism -fmmc-uniqueness %s 2>%t.mmc >/dev/null
% RUN: FileCheck --check-prefix=MMC --implicit-check-not=error: %s < %t.mmc
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module fields.
:- interface.
:- type t ---> t(int).
:- type r ---> r(f :: t, g :: t).
:- pred taken(r::di, t::uo) is det.
:- pred taken_explicit(r::di, t::uo) is det.
:- pred taken_then_used(r::di, t::uo) is det.
:- pred taken_then_other(r::di, t::uo, t::uo) is det.
:- pred taken_then_other_explicit(r::di, t::uo, t::uo) is det.
:- pred updated(r::di, t::di, r::uo) is det.
:- pred updated_in_place(r::di, r::uo) is det.
:- pred updated_value_used(r::di, t::di, r::uo) is det.
:- implementation.
:- import_module int.
:- pred updu(int::in, t::di, t::uo) is det.
updu(Y, t(X), t(X + Y)).
:- pred g(t::in) is det.
g(_).
:- pred gr(r::in) is det.
gr(_).

taken(R0, T) :- updu(1, R0 ^ f, T).
taken_explicit(R0, T) :- R0 = r(F, _), updu(1, F, T).
taken_then_used(R0, T) :- F = R0 ^ f, updu(1, F, T), gr(R0).
% ERR-DAG: fields.m:[[@LINE-1]]:39: error: in the clause for taken_then_used(di, uo): mode error in argument 2 of the call to pred fields.updu/3: F has inst ground, but unique is expected
% MMC-DAG: fields.m:[[@LINE-2]]:39: error: in the clause for taken_then_used(di, uo): mode error in argument 2 of the call to pred fields.updu/3: F has inst ground, but unique is expected
taken_then_other(R0, T, G) :- F = R0 ^ f, updu(1, F, T), G = R0 ^ g.
% ERR-DAG: fields.m:[[@LINE-1]]:43: error: in the clause for taken_then_other(di, uo, uo): mode error in argument 2 of the call to pred fields.updu/3: F has inst ground, but unique is expected
% MMC-DAG: fields.m:[[@LINE-2]]:43: error: in the clause for taken_then_other(di, uo, uo): mode error in argument 2 of the call to pred fields.updu/3: F has inst ground, but unique is expected
taken_then_other_explicit(R0, T, G) :-
    R0 = r(F, _), updu(1, F, T), R0 = r(_, G).
% ERR-DAG: fields.m:[[@LINE-1]]:19: error: in the clause for taken_then_other_explicit(di, uo, uo): mode error in argument 2 of the call to pred fields.updu/3: F has inst ground, but unique is expected
% MMC-DAG: fields.m:[[@LINE-2]]:19: error: in the clause for taken_then_other_explicit(di, uo, uo): mode error in argument 2 of the call to pred fields.updu/3: F has inst ground, but unique is expected
updated(R0, T, R) :- R = R0 ^ g := T.
updated_in_place(R0, R) :- F0 = R0 ^ f, updu(1, F0, F), R = R0 ^ f := F.
% ERR-DAG: fields.m:[[@LINE-1]]:41: error: in the clause for updated_in_place(di, uo): mode error in argument 2 of the call to pred fields.updu/3: F0 has inst ground, but unique is expected
% MMC-DAG: fields.m:[[@LINE-2]]:41: error: in the clause for updated_in_place(di, uo): mode error in argument 2 of the call to pred fields.updu/3: F0 has inst ground, but unique is expected
updated_value_used(R0, T, R) :- R = R0 ^ g := T, g(T).
% ERR-DAG: fields.m:[[@LINE-1]]:50: error: in the clause for updated_value_used(di, di, uo): mode error: argument 3, R, did not get sufficiently instantiated: its final inst is unique(r(unique, ground)), but unique is expected
% MMC-DAG: fields.m:[[@LINE-2]]:50: error: in the clause for updated_value_used(di, di, uo): mode error: argument 3, R, did not get sufficiently instantiated: its final inst is unique(r(unique, ground)), but unique is expected
