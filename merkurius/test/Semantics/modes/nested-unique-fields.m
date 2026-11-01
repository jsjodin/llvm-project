% Unique fields of unique terms in the first phase of unique mode
% analysis: taking them out, clobbering them and putting them back, with
% the sharing that a deconstruction, a construction or an `in' argument
% makes, as in mmc (see the design document, "Unique modes, the first
% phase"). A term with two copies of one argument has both shared, and a
% call's error comes before that of a head unification waiting for its
% output.
% These are mmc's uniqueness rules (-fmmc-uniqueness); by default, a
% value that is copied, such as a number, is unique wherever it is
% (../uniqueness/copied.m).
%
% RUN: not merkc -fc1 -fdebug-dump-modes -fmmc-uniqueness %s 2>%t.err >/dev/null
% RUN: FileCheck --implicit-check-not=error: %s < %t.err
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module nested_unique_fields.
:- interface.
:- import_module list.
:- type t ---> t(int).
:- type p ---> p(t, t).
:- type q ---> q(p, int) ; r.
:- inst up == unique(p(unique, unique)).
:- inst uq == unique(q(up, ground) ; r).
:- inst ul == unique([] ; [unique | ul]).
:- inst hp == unique(p(ground, unique)).
:- inst mup == mostly_unique(p(mostly_unique, unique)).
:- mode dip == up >> dead.
:- mode uop == free >> up.

% Accepted.
:- pred switch_rebuild(q::(uq >> dead), q::(free >> uq)) is det.
:- pred list_update(list(t)::(ul >> dead), list(t)::(free >> ul)) is det.
:- pred condition_rebuild(q::(uq >> dead), q::(free >> uq)) is det.
:- pred shared_field(p::(hp >> dead), p::(free >> hp)) is det.
:- pred constant_field(p::dip, p::uop) is semidet.
:- pred taken_twice(p::dip, p::uo) is det.
:- pred rebuilt_twice(p::dip, p::uop) is det.
:- pred unique_test(p::dip, p::dip, p::uo) is semidet.
% Rejected.
:- pred clobbered_term_used(p::dip, p::uop, p::out) is det.
:- pred field_read(p::dip, p::uop) is det.
:- pred read_then_clobbered(p::dip, t::uo) is det.
:- pred ground_field(p::in, t::uo) is det.
:- pred argument_used(t::di, t::di, p::uop, t::out) is det.
:- pred deconstructed_twice(p::dip, t::uo, t::uo) is det.
:- pred shared_field_out(p::(hp >> dead), p::uop) is det.
:- pred mostly_field(p::(mup >> mostly_dead), p::uop) is det.
:- pred tested_field(p::dip, t::uo) is semidet.
:- pred captured_field(p::dip, p::uo) is det.
:- pred twice(t::di, p::uo) is det.
:- pred head_waits(t::mdi, p::out) is det.
:- pred head_clobbers(t::in, p::out) is det.

:- implementation.
:- import_module int.

:- pred updu(int::in, t::di, t::uo) is det.
updu(Y, t(X), t(X + Y)).
:- pred gin(t::in) is det.
gin(_).
:- pred pdi(p::di, p::uo) is det.
pdi(P, P).

switch_rebuild(Q0, Q) :-
    ( Q0 = q(p(A0, B), N), updu(N, A0, A), Q = q(p(A, B), N)
    ; Q0 = r, Q = r
    ).
list_update(L0, L) :-
    ( L0 = [], L = []
    ; L0 = [H0 | T0], updu(1, H0, H), list_update(T0, T), L = [H | T]
    ).
condition_rebuild(Q0, Q) :-
    ( if Q0 = q(P0, N) then
        P0 = p(A0, B), updu(N, A0, A), Q = q(p(A, B), N)
    else
        Q = r
    ).
shared_field(P0, P) :- P0 = p(A, B0), updu(1, B0, B), P = p(A, B).
constant_field(P0, P) :- P0 = p(t(1), B0), updu(1, B0, B), P = p(t(2), B).
taken_twice(P0, P) :- P0 = p(A0, B0), Q = p(B0, A0), Q = p(C, D), P = p(C, D).
rebuilt_twice(P0, P) :- P0 = p(A0, B), updu(1, A0, A), P = p(A, B).
unique_test(P0, Q0, P) :- P0 = Q0, pdi(P0, P).

clobbered_term_used(P0, P, Q) :- P0 = p(A0, B), updu(1, A0, A), P = p(A, B), Q = P0.
% CHECK: nested-unique-fields.m:[[@LINE-1]]:49: error: in the clause for clobbered_term_used((nested_unique_fields.up >> builtin.dead), out(nested_unique_fields.up), out): mode error in argument 2 of the call to pred nested_unique_fields.updu/3: A0 has inst ground, but unique is expected
field_read(P0, P) :- P0 = p(A0, _), gin(A0), P = P0.
% CHECK: nested-unique-fields.m:[[@LINE-1]]:46: error: in the clause for field_read((nested_unique_fields.up >> builtin.dead), out(nested_unique_fields.up)): mode error: argument 2, P, did not get sufficiently instantiated: its final inst is unique(p(ground, unique)), but nested_unique_fields.up is expected
read_then_clobbered(P0, A) :- P0 = p(A0, _), gin(A0), updu(1, A0, A).
% CHECK: nested-unique-fields.m:[[@LINE-1]]:55: error: in the clause for read_then_clobbered((nested_unique_fields.up >> builtin.dead), uo): mode error in argument 2 of the call to pred nested_unique_fields.updu/3: A0 has inst ground, but unique is expected
ground_field(P0, A) :- P0 = p(A, _).
% CHECK: nested-unique-fields.m:[[@LINE-1]]:29: error: in the clause for ground_field(in, uo): mode error: argument 2, A, did not get sufficiently instantiated: its final inst is ground, but unique is expected
argument_used(A, B, P, A) :- P = p(A, B).
% CHECK: nested-unique-fields.m:[[@LINE-1]]:34: error: in the clause for argument_used(di, di, out(nested_unique_fields.up), out): mode error: argument 3, P, did not get sufficiently instantiated: its final inst is unique(p(ground, unique)), but nested_unique_fields.up is expected
deconstructed_twice(P0, A, B) :- P0 = p(A, _), P0 = p(_, B).
% CHECK: nested-unique-fields.m:[[@LINE-1]]:53: error: in the clause for deconstructed_twice((nested_unique_fields.up >> builtin.dead), uo, uo): mode error: argument 2, A, did not get sufficiently instantiated: its final inst is ground, but unique is expected
shared_field_out(P0, P) :- P0 = p(A, B0), updu(1, B0, B), P = p(A, B).
% CHECK: nested-unique-fields.m:[[@LINE-1]]:63: error: in the clause for shared_field_out((nested_unique_fields.hp >> builtin.dead), out(nested_unique_fields.up)): mode error: argument 2, P, did not get sufficiently instantiated: its final inst is unique(p(ground, unique)), but nested_unique_fields.up is expected
mostly_field(P0, P) :- P0 = p(A0, B), updu(1, A0, A), P = p(A, B).
% CHECK: nested-unique-fields.m:[[@LINE-1]]:39: error: in the clause for mostly_field((nested_unique_fields.mup >> builtin.mostly_dead), out(nested_unique_fields.up)): mode error in argument 2 of the call to pred nested_unique_fields.updu/3: A0 has inst mostly_unique, but unique is expected
tested_field(P0, A) :- P0 = p(A0, _), A0 = t(1), updu(1, A0, A).
% CHECK: nested-unique-fields.m:[[@LINE-1]]:50: error: in the clause for tested_field((nested_unique_fields.up >> builtin.dead), uo): mode error in argument 2 of the call to pred nested_unique_fields.updu/3: A0 has inst unique(t(ground)), but unique is expected
captured_field(P0, P) :-
    P0 = p(A0, B), F = (pred(X::out) is det :- X = A0), F(A1), P = p(A1, B).
% CHECK: nested-unique-fields.m:[[@LINE-1]]:68: error: in the clause for captured_field((nested_unique_fields.up >> builtin.dead), uo): mode error: argument 2, P, did not get sufficiently instantiated: its final inst is unique(p(ground, unique)), but unique is expected
twice(A, P) :- P = p(A, A).
% CHECK: nested-unique-fields.m:[[@LINE-1]]:20: error: in the clause for twice(di, uo): mode error: argument 2, P, did not get sufficiently instantiated: its final inst is unique(p(ground, ground)), but unique is expected
head_waits(T0, p(A, A)) :- updu(1, T0, A).
% CHECK: nested-unique-fields.m:[[@LINE-1]]:28: error: in the clause for head_waits(mdi, out): mode error in argument 2 of the call to pred nested_unique_fields.updu/3: T0 has inst mostly_unique, but unique is expected
head_clobbers(T0, p(A, A)) :- updu(1, T0, A).
% CHECK: nested-unique-fields.m:[[@LINE-1]]:31: error: in the clause for head_clobbers(in, out): unique-mode error in argument 2 of the call to pred nested_unique_fields.updu/3: the called procedure would clobber its argument, but T0 is still live
