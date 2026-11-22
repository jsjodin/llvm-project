% Code generation does not do all of open terms yet (docs/INPROGRESS.md,
% Open types, step 4): a construction of an open term and a get_structure
% wait for step 4(b), and a closed(T) value that may be unbound, as a fresh
% variable or in a full unification, for step 4(c). Each is reported as not
% yet implemented. Full unification, fresh variables and calls of open's
% predicates are generated (open-terms.m).
%
% RUN: not merkc -fc1 -fopen-types -emit-merk %s -o /dev/null 2>&1 \
% RUN:   | FileCheck --implicit-check-not=error: %s

:- module open_types_not_yet.
:- interface.
:- import_module open, list.
:- pred make(open(list(int))::oa) is det.
:- pred get(open(list(int))::ia, open(int)::oa) is semidet.
:- pred same_closed(closed(int)::ia, closed(int)::ia) is semidet.
:- pred fresh_closed(closed(int)::oa) is det.
:- pred pass(open(list(int))::oa) is det.
:- pred hold(closed(int)::ia) is det.
:- implementation.

make([_ | _]).
% CHECK: [[@LINE-1]]:6: error: not yet implemented: code generation for constructions and deconstructions of open terms

get([X | _], X).
% CHECK: [[@LINE-1]]:5: error: not yet implemented: code generation for get_structure on open terms

same_closed(X, X).
% CHECK: [[@LINE-1]]:16: error: not yet implemented: code generation for closed(T) values that may be unbound

fresh_closed(X) :- hold(X), hold(X).
% CHECK: [[@LINE-1]]:20: error: not yet implemented: code generation for closed(T) values that may be unbound

hold(_).

pass(X) :- make(X).
