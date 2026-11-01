% Unique values needed on backtracking in unique terms' fields, in the
% second phase of unique mode analysis: a field taken out before or after
% a goal that can succeed more than once, a term rebuilt or destroyed
% after one, terms two deep, a unique list, disjunctions, a lambda
% expression's nonlocal field, commits, and the if-then-else rules, which
% by default find a field that mmc's (-fmmc-uniqueness) miss.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: %s < %t.err
% RUN: not merkc -fc1 -fdebug-dump-determinism -fmmc-uniqueness %s 2>%t.mmc >/dev/null
% RUN: FileCheck --check-prefix=MMC --implicit-check-not=error: %s < %t.mmc
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module nested.
:- interface.
:- import_module list.
:- type t ---> t(int).
:- type p ---> p(t, t).
:- type pi ---> pi(t, t, int).
:- type pp ---> pp(p, int).
:- inst ut == unique(t(ground)).
:- inst up == unique(p(unique, unique)).
:- inst upn == unique(p(ut, ut)).
:- inst upi == unique(pi(unique, unique, ground)).
:- inst upp == unique(pp(up, ground)).
:- inst ul == unique([] ; [unique | ul]).
:- inst mup == mostly_unique(p(mostly_unique, mostly_unique)).
:- mode dip == up >> dead.
:- mode uop == free >> up.
:- mode mdip == mup >> mostly_dead.
:- mode muop == free >> mup.
:- mode uopi == free >> upi.

:- pred field_before_multi(p::dip, t::uo) is multi.
:- pred field_in_disjunction(p::dip, t::uo) is multi.
:- pred field_clobbered_in_condition(p::dip, p::uop) is det.
:- pred field_after_committed(p::dip, t::uo) is cc_multi.
:- pred field_out_mostly(p::mdip, t::muo) is multi.
:- pred rebuilt_after_multi(p::dip, pi::uopi) is multi.
:- pred updated_then_rebuilt(p::dip, pi::uopi) is multi.
:- pred term_destroyed(p::dip, p::uo, int::out) is multi.
:- pred rebuilt_mostly(p::dip, pi::(free >> mostly_unique(pi(mostly_unique, mostly_unique, ground)))) is multi.
:- pred built_before_taken(p::dip, pi::uopi) is multi.
:- pred unnamed_term_destroyed(p::(unique(p(unique, unique)) >> dead), p::uo, int::out) is multi.
:- pred bound_field_mdi(p::(upn >> dead), t::muo, int::out) is multi.
:- pred bound_field_muo(p::(upn >> dead), t::muo, int::out) is multi.
:- pred inner_field(pp::(upp >> dead), t::uo, int::out) is multi.
:- pred inner_pair_rebuilt(pp::(upp >> dead), pp::(free >> upp), int::out) is multi.
:- pred list_head(list(t)::(ul >> dead), list(t)::(free >> ul), int::out) is multi.
:- pred disjunct_fields(p::(up >> dead), t::uo) is multi.
:- pred field_in_lambda(p::(up >> dead), t::uo, int::out) is multi.
:- pred promised_solutions(p::(up >> dead), t::uo) is det.
:- pred condition_rebuilds(p::(up >> dead), p::(free >> up)) is det.
:- pred after_if_then_else(p::(up >> dead), t::uo) is multi.
:- pred field_mdi_after_multi(p::(up >> dead), t::muo, int::out) is multi.

:- implementation.
:- import_module int.

:- pred updu(int::in, t::di, t::uo) is det.
updu(Y, t(X), t(X + Y)).
:- pred upd(int::in, t::mdi, t::muo) is det.
upd(Y, t(X), t(X + Y)).
:- pred pdi(p::di, p::uo) is det.
pdi(P, P).
:- pred m(int::out) is multi.
m(1).
m(2).
:- pred cm(int::out) is cc_multi.
cm(1).
cm(2).

field_before_multi(P0, A) :- P0 = p(A0, _), m(X), updu(X, A0, A).
% ERR-DAG: nested.m:[[@LINE-1]]:51: error: in the clause for field_before_multi((nested.up >> builtin.dead), uo): mode error in argument 2 of the call to pred nested.updu/3: A0 has inst mostly_unique, but unique is expected
% MMC-DAG: nested.m:[[@LINE-2]]:51: error: in the clause for field_before_multi((nested.up >> builtin.dead), uo): mode error in argument 2 of the call to pred nested.updu/3: A0 has inst mostly_unique, but unique is expected
field_in_disjunction(P0, A) :- P0 = p(A0, _), ( updu(1, A0, A) ; updu(2, A0, A) ).
% ERR-DAG: nested.m:[[@LINE-1]]:49: error: in the clause for field_in_disjunction((nested.up >> builtin.dead), uo): mode error in argument 2 of the call to pred nested.updu/3: A0 has inst mostly_unique, but unique is expected
% MMC-DAG: nested.m:[[@LINE-2]]:49: error: in the clause for field_in_disjunction((nested.up >> builtin.dead), uo): mode error in argument 2 of the call to pred nested.updu/3: A0 has inst mostly_unique, but unique is expected
field_clobbered_in_condition(P0, P) :- ( if P0 = p(A0, _), updu(1, A0, A1), A1 = t(5) then P = p(t(1), t(2)) else P = P0 ).
% ERR-DAG: nested.m:[[@LINE-1]]:60: error: in the clause for field_clobbered_in_condition((nested.up >> builtin.dead), out(nested.up)): mode error in argument 2 of the call to pred nested.updu/3: A0 has inst mostly_unique, but unique is expected
field_after_committed(P0, A) :- P0 = p(A0, _), cm(X), updu(X, A0, A).
field_out_mostly(P0, A) :- P0 = p(A, _), m(_).
rebuilt_after_multi(P0, P) :- P0 = p(A, B), m(X), P = pi(A, B, X).
% ERR-DAG: nested.m:[[@LINE-1]]:55: error: in the clause for rebuilt_after_multi((nested.up >> builtin.dead), out(nested.upi)): mode error: argument 2, P, did not get sufficiently instantiated: its final inst is unique(pi(mostly_unique, mostly_unique, ground)), but nested.upi is expected
% MMC-DAG: nested.m:[[@LINE-2]]:55: error: in the clause for rebuilt_after_multi((nested.up >> builtin.dead), out(nested.upi)): mode error: argument 2, P, did not get sufficiently instantiated: its final inst is unique(pi(mostly_unique, mostly_unique, ground)), but nested.upi is expected
updated_then_rebuilt(P0, P) :- P0 = p(A0, B), updu(1, A0, A), m(X), P = pi(A, B, X).
% ERR-DAG: nested.m:[[@LINE-1]]:73: error: in the clause for updated_then_rebuilt((nested.up >> builtin.dead), out(nested.upi)): mode error: argument 2, P, did not get sufficiently instantiated: its final inst is unique(pi(mostly_unique, mostly_unique, ground)), but nested.upi is expected
% MMC-DAG: nested.m:[[@LINE-2]]:73: error: in the clause for updated_then_rebuilt((nested.up >> builtin.dead), out(nested.upi)): mode error: argument 2, P, did not get sufficiently instantiated: its final inst is unique(pi(mostly_unique, mostly_unique, ground)), but nested.upi is expected
term_destroyed(P0, P, X) :- m(X), pdi(P0, P).
% ERR-DAG: nested.m:[[@LINE-1]]:35: error: in the clause for term_destroyed((nested.up >> builtin.dead), uo, out): mode error in argument 1 of the call to pred nested.pdi/2: P0 has inst mostly_unique(p(mostly_unique, mostly_unique)), but unique is expected
% MMC-DAG: nested.m:[[@LINE-2]]:35: error: in the clause for term_destroyed((nested.up >> builtin.dead), uo, out): mode error in argument 1 of the call to pred nested.pdi/2: P0 has inst mostly_unique(p(mostly_unique, mostly_unique)), but unique is expected
rebuilt_mostly(P0, P) :- P0 = p(A, B), m(X), P = pi(A, B, X).
built_before_taken(P0, P) :- m(X), P = pi(A, B, X), P0 = p(A, B).
% ERR-DAG: nested.m:[[@LINE-1]]:40: error: in the clause for built_before_taken((nested.up >> builtin.dead), out(nested.upi)): mode error: argument 2, P, did not get sufficiently instantiated: its final inst is unique(pi(mostly_unique, mostly_unique, ground)), but nested.upi is expected
% MMC-DAG: nested.m:[[@LINE-2]]:40: error: in the clause for built_before_taken((nested.up >> builtin.dead), out(nested.upi)): mode error: argument 2, P, did not get sufficiently instantiated: its final inst is unique(pi(mostly_unique, mostly_unique, ground)), but nested.upi is expected
unnamed_term_destroyed(P0, P, X) :- m(X), pdi(P0, P).
% ERR-DAG: nested.m:[[@LINE-1]]:43: error: in the clause for unnamed_term_destroyed((unique(p(unique, unique)) >> builtin.dead), uo, out): mode error in argument 1 of the call to pred nested.pdi/2: P0 has inst mostly_unique(p(mostly_unique, mostly_unique)), but unique is expected
% MMC-DAG: nested.m:[[@LINE-2]]:43: error: in the clause for unnamed_term_destroyed((unique(p(unique, unique)) >> builtin.dead), uo, out): mode error in argument 1 of the call to pred nested.pdi/2: P0 has inst mostly_unique(p(mostly_unique, mostly_unique)), but unique is expected
bound_field_mdi(P0, A, X) :- P0 = p(A0, _), m(X), upd(X, A0, A).
% ERR-DAG: nested.m:[[@LINE-1]]:51: error: in the clause for bound_field_mdi((nested.upn >> builtin.dead), muo, out): mode error in argument 2 of the call to pred nested.upd/3: A0 has inst unique(t(ground)), but mostly_unique is expected
% MMC-DAG: nested.m:[[@LINE-2]]:51: error: in the clause for bound_field_mdi((nested.upn >> builtin.dead), muo, out): mode error in argument 2 of the call to pred nested.upd/3: A0 has inst unique(t(ground)), but mostly_unique is expected
bound_field_muo(P0, A, X) :- P0 = p(A, _), m(X).
% ERR-DAG: nested.m:[[@LINE-1]]:44: error: in the clause for bound_field_muo((nested.upn >> builtin.dead), muo, out): mode error: argument 2, A, had the wrong inst: its final inst is unique(t(ground)), but mostly_unique is expected
% MMC-DAG: nested.m:[[@LINE-2]]:44: error: in the clause for bound_field_muo((nested.upn >> builtin.dead), muo, out): mode error: argument 2, A, had the wrong inst: its final inst is unique(t(ground)), but mostly_unique is expected
inner_field(Q0, A, X) :- Q0 = pp(P0, _), m(X), P0 = p(A0, _), updu(X, A0, A).
% ERR-DAG: nested.m:[[@LINE-1]]:63: error: in the clause for inner_field((nested.upp >> builtin.dead), uo, out): mode error in argument 2 of the call to pred nested.updu/3: A0 has inst mostly_unique, but unique is expected
% MMC-DAG: nested.m:[[@LINE-2]]:63: error: in the clause for inner_field((nested.upp >> builtin.dead), uo, out): mode error in argument 2 of the call to pred nested.updu/3: A0 has inst mostly_unique, but unique is expected
inner_pair_rebuilt(Q0, Q, X) :- Q0 = pp(p(A, B), N), m(X), Q = pp(p(A, B), N + X).
% ERR-DAG: nested.m:[[@LINE-1]]:64: error: in the clause for inner_pair_rebuilt((nested.upp >> builtin.dead), out(nested.upp), out): mode error: argument 2, Q, had the wrong inst: its final inst is unique(pp(unique(p(mostly_unique, mostly_unique)), unique)), but nested.upp is expected
% MMC-DAG: nested.m:[[@LINE-2]]:64: error: in the clause for inner_pair_rebuilt((nested.upp >> builtin.dead), out(nested.upp), out): mode error: argument 2, Q, had the wrong inst: its final inst is unique(pp(unique(p(mostly_unique, mostly_unique)), unique)), but nested.upp is expected
list_head(L0, L, X) :- ( L0 = [], L = [], X = 0 ; L0 = [H0 | T], m(X), updu(X, H0, H), L = [H | T] ).
% ERR-DAG: nested.m:[[@LINE-1]]:72: error: in the clause for list_head((nested.ul >> builtin.dead), out(nested.ul), out): mode error in argument 2 of the call to pred nested.updu/3: H0 has inst mostly_unique, but unique is expected
% MMC-DAG: nested.m:[[@LINE-2]]:72: error: in the clause for list_head((nested.ul >> builtin.dead), out(nested.ul), out): mode error in argument 2 of the call to pred nested.updu/3: H0 has inst mostly_unique, but unique is expected
disjunct_fields(P0, A) :- P0 = p(A0, B0), ( updu(1, A0, A) ; updu(2, B0, A) ).
% ERR-DAG: nested.m:[[@LINE-1]]:45: error: in the clause for disjunct_fields((nested.up >> builtin.dead), uo): mode error in argument 2 of the call to pred nested.updu/3: A0 has inst mostly_unique, but unique is expected
% MMC-DAG: nested.m:[[@LINE-2]]:45: error: in the clause for disjunct_fields((nested.up >> builtin.dead), uo): mode error in argument 2 of the call to pred nested.updu/3: A0 has inst mostly_unique, but unique is expected
field_in_lambda(P0, A, X) :- P0 = p(A0, _), m(X), F = (pred(R::uo) is det :- updu(X, A0, R)), F(A).
% ERR-DAG: nested.m:[[@LINE-1]]:78: error: in the clause for field_in_lambda((nested.up >> builtin.dead), uo, out): mode error in argument 2 of the call to pred nested.updu/3: A0 has inst ground, but unique is expected
% MMC-DAG: nested.m:[[@LINE-2]]:78: error: in the clause for field_in_lambda((nested.up >> builtin.dead), uo, out): mode error in argument 2 of the call to pred nested.updu/3: A0 has inst ground, but unique is expected
promised_solutions(P0, A) :- promise_equivalent_solutions [X] ( m(X) ), P0 = p(A0, _), updu(X, A0, A).
condition_rebuilds(P0, P) :- ( if P0 = p(A0, B), updu(1, A0, A), A = t(3) then P = p(A, B) else P = P0 ).
% ERR-DAG: nested.m:[[@LINE-1]]:97: error: in the clause for condition_rebuilds((nested.up >> builtin.dead), out(nested.up)): mode error: argument 2, P, did not get sufficiently instantiated: its final inst is unique(p(unique(t(ground)), unique)), but nested.up is expected
% MMC-DAG: nested.m:[[@LINE-2]]:97: error: in the clause for condition_rebuilds((nested.up >> builtin.dead), out(nested.up)): mode error: argument 2, P, did not get sufficiently instantiated: its final inst is unique(p(unique(t(ground)), unique)), but nested.up is expected
after_if_then_else(P0, A) :- ( if m(X), X > 1 then Y = X else Y = 0 ), P0 = p(A0, _), updu(Y, A0, A).
% ERR-DAG: nested.m:[[@LINE-1]]:87: error: in the clause for after_if_then_else((nested.up >> builtin.dead), uo): mode error in argument 2 of the call to pred nested.updu/3: A0 has inst mostly_unique, but unique is expected
field_mdi_after_multi(P0, A, X) :- m(X), P0 = p(A0, _), upd(X, A0, A).
