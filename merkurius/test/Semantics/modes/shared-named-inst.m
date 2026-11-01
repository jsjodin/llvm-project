% A unique value passed to an `in' argument and used again is shared
% after the call, whatever its inst. mmc keeps some named insts unique
% there, as `up' below, which is unsound: `aliased' destroys P0 while Q
% refers to it (see the design document's register). So this is not
% compared with mmc.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>%t.err >/dev/null
% RUN: FileCheck --implicit-check-not=error: %s < %t.err

:- module shared_named_inst.
:- interface.
:- type t ---> t(int).
:- type p ---> p(t, t).
:- inst up == unique(p(unique, unique)).
:- pred aliased(p::(up >> dead), p::uo, p::out) is det.
:- pred read_then_destroyed(p::(up >> dead), p::uo) is det.
:- pred read_then_taken(p::(up >> dead), t::uo) is det.
:- implementation.
:- pred pin(p::in) is det.
pin(_).
:- pred cp(p::in, p::out) is det.
cp(P, P).
:- pred pdi(p::di, p::uo) is det.
pdi(P, P).
aliased(P0, P, Q) :- cp(P0, Q), pdi(P0, P).
% CHECK: shared-named-inst.m:[[@LINE-1]]:33: error: in the clause for aliased((shared_named_inst.up >> builtin.dead), uo, out): mode error in argument 1 of the call to pred shared_named_inst.pdi/2: P0 has inst bound(p(ground, ground)), but unique is expected
read_then_destroyed(P0, P) :- pin(P0), pdi(P0, P).
% CHECK: shared-named-inst.m:[[@LINE-1]]:40: error: in the clause for read_then_destroyed((shared_named_inst.up >> builtin.dead), uo): mode error in argument 1 of the call to pred shared_named_inst.pdi/2: P0 has inst bound(p(ground, ground)), but unique is expected
read_then_taken(P0, A) :- pin(P0), P0 = p(A, _).
% CHECK: shared-named-inst.m:[[@LINE-1]]:41: error: in the clause for read_then_taken((shared_named_inst.up >> builtin.dead), uo): mode error: argument 2, A, did not get sufficiently instantiated: its final inst is ground, but unique is expected
