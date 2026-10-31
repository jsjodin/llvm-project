% Mode inference stops at its iteration limit, 30 passes by default
% (-fmode-inference-iteration-limit), as mmc's does, which reports it
% without a line: here the inferred modes of len/2 for a partially
% instantiated list grow with each pass, since only ground insts are
% normalised (Overton, 2003, section 3.4.2, "Limitations of Mode
% Inference"); mmc's grow alike.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=because: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module inference_limit.
:- interface.

:- pred f(int::out) is semidet.

:- implementation.
:- import_module int, list.

f(N) :-
    len([1 | _], N).

len([], 0).
% CHECK: [[@LINE-1]]:1: error: mode inference did not reach a fixpoint in 30 passes
% CHECK: [[@LINE-2]]:5: warning: the unification of HeadVar__1 and V_0 cannot succeed: HeadVar__1 has inst unique('[|]'(unique(1), free)), and V_0 has inst unique([])
len([_ | T], N) :-
    len(T, M),
    N = M + 1.
