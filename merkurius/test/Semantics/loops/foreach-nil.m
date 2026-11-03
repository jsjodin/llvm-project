% A foreach's translation unifies its list with [] and [X | T], and its
% switch on the list is complete, so the loop is det, even in a module
% that writes no [] of its own: type checking makes the signatures of all
% of the list type's constructors, which mode analysis needs to see that
% the switch cannot fail.
%
% RUN: merkc -fc1 -fdo-loops -fdebug-dump-determinism %s 2>&1 \
% RUN:   | FileCheck --implicit-check-not=error: %s

:- module foreach_nil.
:- interface.
:- import_module list.
:- pred sum(list(int)::in, int::out) is det.
:- implementation.
:- import_module int.

sum(Xs, S) :-
    ( foreach(X, Xs), fromto(0, S0, S1, S) do S1 = S0 + X ).
% CHECK: [[@LINE-1]]:7 pred foreach_nil.sum__loop_[[@LINE-1]]_7/3 inferred mode 0: sum__loop_[[@LINE-1]]_7(in, di, uo), inferred det
% CHECK-NEXT: [[@LINE-2]]:7 switch on HeadVar__1, cannot fail [det]
