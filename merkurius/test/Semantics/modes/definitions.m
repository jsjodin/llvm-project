% Errors in inst and mode definitions (reference manual, "Insts, modes,
% and mode definitions"): reserved names, an inst variable that is not a
% parameter, which leaves the definition undefined, as in mmc, and
% definitions that expand to themselves.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module definitions.
:- interface.

:- inst ground == bound(a).
% CHECK: [[@LINE-1]]:9: error: the inst name ground is reserved for the Mercury implementation
:- mode (>>) == in.
% CHECK: [[@LINE-1]]:10: error: the mode name >> is reserved for the Mercury implementation
:- inst bad(I) == bound(f(J)).
% CHECK: [[@LINE-1]]:9: error: inst parameter J is not a parameter of the inst bad/1
:- mode badm(I) == J >> I.
% CHECK: [[@LINE-1]]:9: error: inst parameter J is not a parameter of the mode badm/1
:- inst a == b.
% CHECK: [[@LINE-1]]:9: error: the inst a/0 expands to an inst containing itself, through b/0
% CHECK: [[@LINE+1]]:9: because: the definition of b/0 is here
:- inst b == a.
:- inst self == self.
% CHECK: [[@LINE-1]]:9: error: the inst self/0 expands to an inst containing itself
:- mode m1 == m2.
% CHECK: [[@LINE-1]]:9: error: the mode m1/0 expands to a mode containing itself, through m2/0
% CHECK: [[@LINE+1]]:9: because: the definition of m2/0 is here
:- mode m2 == m1.
:- mode m3(I) == m3(I).
% CHECK: [[@LINE-1]]:9: error: the mode m3/1 expands to a mode containing itself

:- type t ---> f(int).
:- pred p(t::in(bad(ground))) is det.
% CHECK: [[@LINE-1]]:17: error: undefined inst bad/1
:- pred q(t::badm(ground)) is det.
% CHECK: [[@LINE-1]]:14: error: undefined mode badm/1

:- implementation.
p(_).
q(_).
