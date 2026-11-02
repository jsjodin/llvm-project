% Of a committed choice call in a goal with no outputs, whose first
% solution is the only one wanted, the reasons that all are needed, when
% something in the goal asks for them, are the goals after it that can
% fail too, as in mmc, even one whose determinism is inferred. Not
% compared with mmc, which labels the if-then-else nondet, where merkc
% labels it semidet, as one with no outputs; the errors and reasons are
% at mmc's lines.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: --implicit-check-not=because: %s < %t.err

:- module no_outputs.
:- interface.
:- pred no_outputs(int::in, int::out) is multi.
% ERR-DAG: no-outputs.m:[[@LINE-1]]:9: error: the determinism declaration of no_outputs(in, out) is not satisfied: it is declared multi, but inferred semidet
:- implementation.
:- import_module int.
:- pred cm(int::out) is cc_multi.
cm(1).
cm(2).
:- pred sd(int::in).
sd(X) :- X > 0.
no_outputs(A, B) :-
    ( if cm(X) then X > 1 else true ),
% ERR-DAG: no-outputs.m:[[@LINE-1]]:21: because: the call to >(in, in) can fail
% ERR-DAG: no-outputs.m:[[@LINE-2]]:10: error: the call to cm(out) with determinism cc_multi is in a context that requires all solutions
% ERR-DAG: no-outputs.m:[[@LINE-3]]:21: because: the call to >(in, in) can fail
    sd(A),
% ERR-DAG: no-outputs.m:[[@LINE-1]]:5: because: the call to sd(in) can fail
% ERR-DAG: no-outputs.m:[[@LINE-2]]:5: because: the call to sd(in) can fail
    B = A.
