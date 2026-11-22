% Code generation does not do all of open terms yet (docs/INPROGRESS.md,
% Open types, step 4). A closed(T) value that may be unbound holds its open
% word (open-closed.m), except where the callee reads it at a type
% variable, through a closure, or in an ordinary term, which holds T's word.
% An open term that holds
% an array, a higher-order value or an existentially typed constructor,
% which the runtime's walkers stop on, is reported where open's predicates
% or a full unification would walk it; a closed(U) holding one only where
% it is unified, as to_open and from_open share its box. A message with open
% terms in it waits for step 6, which copies it. Each is reported as not
% yet implemented. Full unification, fresh variables, calls of open's
% predicates (open-terms.m), constructions and get_structure
% (open-construct.m) and closed(T) holes (open-closed.m) are generated.
%
% RUN: not merkc -fc1 -fopen-types -emit-merk %s -o /dev/null 2>&1 \
% RUN:   | FileCheck --implicit-check-not=error: %s

:- module open_types_not_yet.
:- interface.
:- import_module array, io, list, open, process.
:- pred make(open(list(int))::oa) is det.
:- pred get(open(list(int))::ia, open(int)::oa) is semidet.
:- pred same_closed(closed(int)::ia, closed(int)::ia) is semidet.
:- pred fresh_closed(closed(int)::oa) is det.
:- pred pass(open(list(int))::oa) is det.
:- pred hold(closed(int)::ia) is det.
:- implementation.

make([_ | _]).

get([X | _], X).

:- type bundle ---> bundle(closed(list(int)), int).
:- pred closed_field(open(bundle)::ia, open(int)::oa) is semidet.
closed_field(bundle(_, N), N).

same_closed(X, X).

fresh_closed(X) :- hold(X), hold(X).

:- pred poly(T::ia) is det.
poly(_).
:- pred to_poly(closed(int)::ia) is det.
to_poly(X) :- poly(X).
% CHECK: [[@LINE-1]]:15: error: not yet implemented: code generation for passing a closed(T) value that may be unbound where the callee has a type variable

:- pred in_list(closed(int)::ia, list(closed(int))::oa) is det.
in_list(X, [X]).
% CHECK: [[@LINE-1]]:12: error: not yet implemented: code generation for terms of type list.list(open.closed(int)) holding closed(T) values that may be unbound

:- pred through(pred(closed(int)), closed(int)).
:- mode through(pred(ia) is det, ia) is det.
through(P, X) :- P(X).
% CHECK: [[@LINE-1]]:18: error: not yet implemented: code generation for higher-order calls with closed(T) arguments that may be unbound


hold(_).

pass(X) :- make(X).

:- type ex ---> some [T] ex(T).
:- type holder ---> holder(closed(array(int))).
:- type fn ---> fn(func(int) = int).

:- pred arrays(open(list(array(int)))::oa) is det.
arrays(to_open([array([1])])).
% CHECK: [[@LINE-1]]:8: error: not yet implemented: code generation for open terms that hold arrays, as values of type open.open(list.list(array.array(int))) can

:- pred closures(open(fn)::oa) is det.
closures(to_open(fn(func(X) = X))).
% CHECK: [[@LINE-1]]:10: error: not yet implemented: code generation for open terms that hold higher-order values, as values of type open.open(open_types_not_yet.fn) can

:- pred existentials(open(ex)::oa) is det.
existentials(to_open('new ex'(1))).
% CHECK: [[@LINE-1]]:14: error: not yet implemented: code generation for open terms that hold existentially typed constructors, as values of type open.open(open_types_not_yet.ex) can

:- pred closed_shared(open(holder)::oa) is det.
closed_shared(to_open(holder(closed(array([1]))))).

:- pred closed_unified(open(holder)::ia, open(holder)::ia) is semidet.
closed_unified(X, X).
% CHECK: [[@LINE-1]]:19: error: not yet implemented: code generation for open terms that hold arrays, as values of type open.open(open_types_not_yet.holder) can

:- pred sends(mailbox(open(list(int)))::in, io::di, io::uo) is det.
sends(M, !IO) :- send(M, to_open([1]), !IO).
% CHECK: [[@LINE-1]]:18: error: not yet implemented: code generation for messages with open terms in them, such as values of type process.mailbox(open.open(list.list(int)))
