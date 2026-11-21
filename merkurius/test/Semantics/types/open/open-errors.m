% Errors of open types (docs/INPROGRESS.md, Open types): closed(T) may not
% hold an open type, at any depth; a clause that fails both with and without
% the open twins reports the second pass's errors when they come later in
% the clause, as the twins removed the first pass's first error; and the
% library module open needs -fopen-types.
%
% RUN: not merkc -fc1 -fopen-types -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --check-prefix=FLAG %s

:- module open_errors.
:- interface.
:- import_module open, list.
% FLAG: [[@LINE-1]]:18: error: the module open, of open types, needs -fopen-types
:- type bad ---> bad(closed(open(int))).
% CHECK: [[@LINE-1]]:22: error: closed(T) holds a ground value, so its argument may not have an open type in it
:- type bad2 ---> bad2(closed(list(open(int)))).
% CHECK: [[@LINE-1]]:24: error: closed(T) holds a ground value, so its argument may not have an open type in it
:- pred r(open(list(int))::oa, int::out) is det.
:- implementation.
r(X, Y) :- X = [Y], Y = "a".
% CHECK: [[@LINE-1]]:25: error: type error in unification of variable Y and `"a"': variable Y has type int, but `"a"' has type string
