% A type variable quantified twice is an error. mmc crashes on it
% (map.det_insert: key already present), so it is not compared (see the
% register); mmc's inferred declarations can name two existential type
% variables alike, as `some [T, T]`.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s

:- module quantified_twice.
:- interface.
:- some [T, T] pred pair_of(T::out, T::out) is det.
% CHECK: [[@LINE-1]]:4: error: type variable T is quantified twice in the declaration of predicate pair_of/2
:- implementation.
pair_of(1, 2).
