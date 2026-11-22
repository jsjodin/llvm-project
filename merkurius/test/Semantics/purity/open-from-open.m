% from_open is impure, as var/1 is: it can fail on a term with a hole and
% succeed once the hole is bound, so a call must say `impure`, and a pure
% predicate around it must promise purity. An impure goal stays where it is,
% after the goals that bind the term's holes.
%
% RUN: not merkc -fc1 -fopen-types -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s

:- module open_from_open.
:- interface.
:- import_module list, open.
:- pred unannotated(open(list(int))::ia, list(int)::out) is semidet.
:- pred unpromised(open(list(int))::ia, list(int)::out) is semidet.
:- pred promised(open(list(int))::ia, list(int)::out) is semidet.
:- implementation.

% CHECK-DAG: error: pred open_from_open.unannotated/2 is impure, so it must be declared `impure' or promised pure
% CHECK-DAG: error: pred open_from_open.unpromised/2 is impure, so it must be declared `impure' or promised pure

unannotated(X, L) :- from_open(X, L).
% CHECK-DAG: [[@LINE-1]]:22: error: the call of impure pred open.from_open/2 must be annotated `impure'

unpromised(X, L) :- impure from_open(X, L).

promised(X, L) :- promise_pure ( impure from_open(X, L) ).
