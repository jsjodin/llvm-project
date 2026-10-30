% A recursive call that permutes the arguments reaches its fixpoint in the
% second pass, f(T1, T2, T3) = T4. mmc does not see the fixpoint at any
% --type-inference-iteration-limit, though it prints that signature, and
% accepts it declared (see the design document's register); the oracle
% checks that.
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types %s %}

:- module missed_fixpoint.
:- interface.
:- pred main(int::out) is det.
:- implementation.

main(0).
f(H1, _, H3) = H4 :- H4 = f(H3, {H1}, H1).
% CHECK: clause at [[@LINE-1]]:1 of func missed_fixpoint.f/3: inferred func missed_fixpoint.f(T1, T2, T3) = T4
