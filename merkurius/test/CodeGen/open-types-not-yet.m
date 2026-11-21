% Code generation does not handle open terms yet (docs/INPROGRESS.md, Open
% types, steps 2 to 4): a unification of an open-typed variable, a
% get_structure, a full unification, a fresh variable, and a function call
% whose result is open are reported as not yet implemented.
% A procedure that only passes open values on, by calls, is generated.
%
% RUN: not merkc -fc1 -fopen-types -emit-merk %s -o /dev/null 2>&1 \
% RUN:   | FileCheck --implicit-check-not=error: %s

:- module open_types_not_yet.
:- interface.
:- import_module open, list.
:- pred make(open(list(int))::oa) is det.
:- pred get(open(list(int))::ia, open(int)::oa) is semidet.
:- pred same(open(int)::ia, open(int)::ia) is semidet.
:- pred wrap(list(int)::in, open(list(int))::oa) is det.
:- pred pass(open(list(int))::oa) is det.
:- implementation.

make([_ | _]).
% CHECK: [[@LINE-1]]:6: error: not yet implemented: code generation for unifications of open terms

get([X | _], X).
% CHECK: [[@LINE-1]]:5: error: not yet implemented: code generation for unifications of open terms

same(X, X).
% CHECK: [[@LINE-1]]:9: error: not yet implemented: code generation for unifications of open terms

wrap(L, to_open(L)).
% CHECK: [[@LINE-1]]:9: error: not yet implemented: code generation for unifications of open terms

pass(X) :- make(X).
