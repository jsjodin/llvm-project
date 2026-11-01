% Values that are copied, not referred to: numbers, characters and
% enumerations, which have no other references however they are used, so
% that by default a shared one is as unique. A unique term whose field of
% such a type has been taken out is still unique: it may be passed to a
% `ui` argument, a `ui` head variable may be read, and the term may then
% be updated (`di`, `mdi`). A field of another type taken out, a string
% or a term, makes the term shared in that part, as in mmc. mmc compares
% copied values as any other (-fmmc-uniqueness), so it rejects all these
% (reference manual, "Limitations of the current implementation": "it is
% also not possible to use unique-input modes").
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: %s < %t.err
% RUN: not merkc -fc1 -fdebug-dump-determinism -fmmc-uniqueness %s 2>%t.mmc >/dev/null
% RUN: FileCheck --check-prefix=MMC --implicit-check-not=error: %s < %t.mmc
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module copied.
:- interface.
:- import_module char.
:- import_module list.
:- type t ---> t(int).
:- type u ---> u(t, int).
:- type e ---> a ; b.
:- type w ---> w(e, float, char).
:- type s ---> s(string).
:- type l ---> l(list(int)).
:- pred read_ui(t::ui, int::out) is det.
:- pred read_second_ui(u::ui, int::out) is det.
:- pred passed_after_read(t::ui, int::out) is det.
:- pred read_then_updated(t::di, t::uo) is det.
:- pred output_then_updated(t::di, t::uo, int::out) is det.
:- pred read_then_passed_and_updated(t::di, t::uo) is det.
:- pred enum_float_char(w::di, w::uo, e::out) is det.
:- pred second_then_updated(u::di, u::uo, int::out) is det.
:- pred read_then_mostly(t::mdi, t::muo, int::out) is det.
:- pred shared_enum(e::di) is det.
:- pred term_field_ui(u::ui, t::out) is det.
:- pred term_field_updated(u::di, u::uo, t::out) is det.
:- pred string_field_updated(s::di, s::uo, string::out) is det.
:- pred list_field_updated(l::di, l::uo, list(int)::out) is det.
:- implementation.
:- import_module int.
:- pred updu(int::in, t::di, t::uo) is det.
updu(Y, t(X), t(X + Y)).
:- pred pass(t::ui) is det.
pass(_).
:- pred mupd(t::mdi, t::muo) is det.
mupd(T, T).
:- pred updv(u::di, u::uo) is det.
updv(U, U).
:- pred updw(w::di, w::uo) is det.
updw(W, W).
:- pred upds(s::di, s::uo) is det.
upds(S, S).
:- pred updl(l::di, l::uo) is det.
updl(L, L).
:- pred eat(e::di) is det.
eat(_).

read_ui(T, X) :- T = t(X).
% MMC-DAG: copied.m:[[@LINE-1]]:22: error: in the clause for read_ui(ui, out): mode error: argument 1, T, did not get sufficiently instantiated: its final inst is unique(t(ground)), but unique is expected
read_second_ui(U, X) :- U = u(_, X).
% MMC-DAG: copied.m:[[@LINE-1]]:29: error: in the clause for read_second_ui(ui, out): mode error: argument 1, U, did not get sufficiently instantiated: its final inst is unique(u(unique, ground)), but unique is expected
passed_after_read(T, X) :- T = t(X), pass(T).
% MMC-DAG: copied.m:[[@LINE-1]]:38: error: in the clause for passed_after_read(ui, out): mode error in argument 1 of the call to pred copied.pass/1: T has inst unique(t(ground)), but unique is expected
read_then_updated(T0, T) :- T0 = t(X), updu(X, T0, T).
% MMC-DAG: copied.m:[[@LINE-1]]:40: error: in the clause for read_then_updated(di, uo): mode error in argument 2 of the call to pred copied.updu/3: T0 has inst unique(t(ground)), but unique is expected
output_then_updated(T0, T, X) :- T0 = t(X), updu(1, T0, T).
% MMC-DAG: copied.m:[[@LINE-1]]:45: error: in the clause for output_then_updated(di, uo, out): mode error in argument 2 of the call to pred copied.updu/3: T0 has inst unique(t(ground)), but unique is expected
read_then_passed_and_updated(T0, T) :- T0 = t(X), pass(T0), updu(X, T0, T).
% MMC-DAG: copied.m:[[@LINE-1]]:51: error: in the clause for read_then_passed_and_updated(di, uo): mode error in argument 1 of the call to pred copied.pass/1: T0 has inst unique(t(ground)), but unique is expected
enum_float_char(W0, W, E) :- W0 = w(E, _, _), updw(W0, W).
% MMC-DAG: copied.m:[[@LINE-1]]:47: error: in the clause for enum_float_char(di, uo, out): mode error in argument 1 of the call to pred copied.updw/2: W0 has inst unique(w(ground, unique, unique)), but unique is expected
second_then_updated(U0, U, N) :- U0 = u(_, N), updv(U0, U).
% MMC-DAG: copied.m:[[@LINE-1]]:48: error: in the clause for second_then_updated(di, uo, out): mode error in argument 1 of the call to pred copied.updv/2: U0 has inst unique(u(unique, ground)), but unique is expected
read_then_mostly(T0, T, X) :- T0 = t(X), mupd(T0, T).
% MMC-DAG: copied.m:[[@LINE-1]]:42: error: in the clause for read_then_mostly(mdi, muo, out): mode error in argument 1 of the call to pred copied.mupd/2: T0 has inst mostly_unique(t(ground)), but mostly_unique is expected
shared_enum(E) :- F = E, eat(E), eat(F).
% MMC-DAG: copied.m:[[@LINE-1]]:26: error: in the clause for shared_enum(di): mode error in argument 1 of the call to pred copied.eat/1: E has inst ground, but unique is expected
term_field_ui(U, A) :- U = u(A, _).
% ERR-DAG: copied.m:[[@LINE-1]]:28: error: in the clause for term_field_ui(ui, out): mode error: argument 1, U, did not get sufficiently instantiated: its final inst is unique(u(ground, unique)), but unique is expected
% MMC-DAG: copied.m:[[@LINE-2]]:28: error: in the clause for term_field_ui(ui, out): mode error: argument 1, U, did not get sufficiently instantiated: its final inst is unique(u(ground, unique)), but unique is expected
term_field_updated(U0, U, A) :- U0 = u(A, _), updv(U0, U).
% ERR-DAG: copied.m:[[@LINE-1]]:47: error: in the clause for term_field_updated(di, uo, out): mode error in argument 1 of the call to pred copied.updv/2: U0 has inst unique(u(ground, unique)), but unique is expected
% MMC-DAG: copied.m:[[@LINE-2]]:47: error: in the clause for term_field_updated(di, uo, out): mode error in argument 1 of the call to pred copied.updv/2: U0 has inst unique(u(ground, unique)), but unique is expected
string_field_updated(S0, S, X) :- S0 = s(X), upds(S0, S).
% ERR-DAG: copied.m:[[@LINE-1]]:46: error: in the clause for string_field_updated(di, uo, out): mode error in argument 1 of the call to pred copied.upds/2: S0 has inst unique(s(ground)), but unique is expected
% MMC-DAG: copied.m:[[@LINE-2]]:46: error: in the clause for string_field_updated(di, uo, out): mode error in argument 1 of the call to pred copied.upds/2: S0 has inst unique(s(ground)), but unique is expected
list_field_updated(L0, L, X) :- L0 = l(X), updl(L0, L).
% ERR-DAG: copied.m:[[@LINE-1]]:44: error: in the clause for list_field_updated(di, uo, out): mode error in argument 1 of the call to pred copied.updl/2: L0 has inst unique(l(ground)), but unique is expected
% MMC-DAG: copied.m:[[@LINE-2]]:44: error: in the clause for list_field_updated(di, uo, out): mode error in argument 1 of the call to pred copied.updl/2: L0 has inst unique(l(ground)), but unique is expected
