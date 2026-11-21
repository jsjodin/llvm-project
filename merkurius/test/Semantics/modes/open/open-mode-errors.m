% Mode and determinism errors of open terms (docs/INPROGRESS.md, Open
% types): binding an existing variable may fail, and so may a get_structure
% of a type with several functors; closed(T) holds a ground value; a
% condition that may bind an open nonlocal needs a purity promise; and
% free does not match any, so a branch may not leave free what another
% binds.
%
% RUN: not merkc -fc1 -fopen-types -fdebug-dump-determinism %s 2>&1 | FileCheck --implicit-check-not=error: %s

:- module open_mode_errors.
:- interface.
:- import_module open, list.
:- pred nil(open(list(int))::ia) is det.
:- pred same(open(int)::ia, open(int)::ia) is det.
:- pred c(int::ia, closed(int)::out) is det.
:- pred neg(open(int)::ia) is semidet.
:- pred merge(int::in, open(int)::oa) is det.
:- implementation.
:- import_module int.

nil(X) :- X = [].
% CHECK: [[@LINE-9]]:9: error: the determinism declaration of nil(ia) is not satisfied: it is declared det, but inferred semidet
% CHECK: [[@LINE-2]]:15: because: the unification of X and []/0 can fail

same(X, X).
% CHECK: [[@LINE-12]]:9: error: the determinism declaration of same(ia, ia) is not satisfied: it is declared det, but inferred semidet
% CHECK: [[@LINE-2]]:9: because: the unification of HeadVar__2 and X can fail

c(X, closed(X)).
% CHECK: [[@LINE-1]]:6: error: in the clause for c(ia, out): mode error in the unification of HeadVar__2 and closed(X): closed(T) holds a ground value, but X has inst any

neg(X) :- \+ X = to_open(1).
% CHECK: [[@LINE-1]]:11: error: in the clause for neg(ia): purity error: the negation must be inside a promise_purity scope, because nonlocal X has inst any and appears in its body

merge(N, X) :- ( if N > 0 then X = to_open(1) else true ).
% CHECK: [[@LINE-1]]:52: error: in the clause for merge(in, oa): mode mismatch in the if-then-else: X is ground in some branches but not others
% CHECK: [[@LINE-2]]:32: because: in this branch, X has inst ground
% CHECK: [[@LINE-3]]:52: because: in this branch, X has inst free
