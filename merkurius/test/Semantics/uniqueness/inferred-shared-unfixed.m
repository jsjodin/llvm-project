% An argument of an inferred procedure that uniqueness checking finds only
% mostly unique where the inferred mode wants it unique is inferred shared
% in a second run (test/CodeGen/inferred-shared.m). Where sharing does not
% fix the error, as here, where the procedure's own call of array.set needs
% the array unique, the second run has as many errors as the first, and the
% first run's message is the one reported: the caller passes A0, which the
% disjunction after it keeps live, so only mostly unique. mmc reports the
% same call, with the procedure's inferred mode.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --implicit-check-not=error: %s < %t.err

:- module inferred_shared_unfixed.
:- interface.
:- import_module array.
:- pred r(array(int)::out) is multi.
:- implementation.
:- import_module int.

r(A) :-
    array.init(3, 0, A0),
    ( K = 1 ; K = 2 ),
    upd(A0, K, A).
% CHECK: [[@LINE-1]]:5: error: in the clause for r(out): mode error in argument 1 of the call to pred inferred_shared_unfixed.upd/3: A0 has inst mostly_unique, but unique is expected

upd(A0, K, A) :-
    array.set(0, K, A0, A).
