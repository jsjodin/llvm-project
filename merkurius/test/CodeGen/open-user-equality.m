% A value of a type with a user-defined equality inside an open term
% (design document, "Open terms as built"): the runtime's unification of
% open terms compares structurally, and cannot call the type's own unify
% entry yet, so it stops at the type's descriptor, marked kUserEquality,
% with a message, rather than answer with the wrong equality.
%
% RUN: merkc -fopen-types %s -o %t
% RUN: not %t 2>&1 | FileCheck %s --match-full-lines
% CHECK: Merkurius runtime: open terms of types with a user-defined equality, such as open_user_equality.set, are not supported yet

:- module open_user_equality.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, open.

:- type set ---> set(list(int)) where equality is set_equals.
:- pred set_equals(set::in, set::in) is semidet.
set_equals(A, B) :-
    promise_equivalent_solutions [X, Y] ( A = set(X), B = set(Y) ),
    list.sort(X, S), list.sort(Y, S).

:- type holder ---> holder(closed(set), int).

:- pred make(set::in, open(holder)::oa) is det.
make(S, holder(closed(S), 1)).

main(!IO) :-
    make(set([1, 2]), A),
    make(set([2, 1]), B),
    promise_pure ( if equal(A, B) then io.write_string("equal\n", !IO)
    else io.write_string("different\n", !IO) ).

:- pred equal(open(T)::ia, open(T)::ia) is semidet.
equal(X, Y) :- X = Y.
