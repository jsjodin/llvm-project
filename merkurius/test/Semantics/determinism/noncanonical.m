% Deconstructions and switches on a variable of a type with a user-defined
% equality (reference manual, "User-defined equality and comparison"): one
% that cannot fail is cc_multi, so it is an error where all solutions are
% wanted, with the goals after it that can fail, and one that can fail is
% an error. An (in, in) unification calls the equality and is semidet. A
% subtype has its base type's equality. As mmc checks them.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --implicit-check-not=error: --implicit-check-not=because: %s < %t.err
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module noncanonical.
:- interface.
:- type unused ---> unused.
:- implementation.
:- import_module list.

:- type set
    --->    set(list(int))
            where equality is set_equals.
:- pred set_equals(set::in, set::in) is semidet.
set_equals(A, B) :-
    promise_equivalent_solutions [X, Y] (
        A = set(X),
        B = set(Y)
    ),
    X = Y.

:- type two
    --->    one(int)
    ;       two(int)
            where equality is two_equals.
:- pred two_equals(two::in, two::in) is semidet.
two_equals(_, _) :- semidet_true.

:- type only_one =< two
    --->    one(int).

% Committed choice in a committed choice procedure, and in a scope that
% promises equivalent solutions.
:- pred elements_cc(set::in, list(int)::out) is cc_multi.
elements_cc(S, L) :- S = set(L).
:- pred elements(set::in, list(int)::out) is det.
elements(S, L) :- promise_equivalent_solutions [L] ( S = set(L) ).
:- pred same(set::in, set::in) is semidet.
same(A, B) :- A = B.
:- pred which_cc(two::in, int::out) is cc_multi.
which_cc(T, N) :- ( T = one(N) ; T = two(N) ).

:- pred elements_det(set::in, list(int)::out) is det.
% CHECK: [[@LINE-1]]:9: error: the determinism declaration of elements_det(in, out) is not satisfied: it is declared det, but inferred multi
elements_det(S, L) :- S = set(L).
% CHECK: [[@LINE-1]]:27: because: the unification of S and set/1 can succeed more than once
% CHECK: [[@LINE-2]]:27: error: the deconstruction of S, whose type noncanonical.set has a user-defined equality, with determinism cc_multi is in a context that requires all solutions

:- pred is_empty(set::in) is semidet.
is_empty(S) :- S = set(L), L = [].
% CHECK: [[@LINE-1]]:20: error: the deconstruction of S, whose type noncanonical.set has a user-defined equality, with determinism cc_multi is in a context that requires all solutions
% CHECK: [[@LINE-2]]:32: because: the unification of L and []/0 can fail

:- pred is_one(two::in) is semidet.
is_one(T) :- T = one(_).
% CHECK: [[@LINE-1]]:18: error: the deconstruction of T can fail, but its type noncanonical.two has a user-defined equality

:- pred which(two::in, int::out) is det.
% CHECK: [[@LINE-1]]:9: error: the determinism declaration of which(in, out) is not satisfied: it is declared det, but inferred multi
which(T, N) :- ( T = one(N) ; T = two(N) ).
% CHECK: [[@LINE-1]]:18: because: the switch on T, whose type has a user-defined equality, can succeed more than once
% CHECK: [[@LINE-2]]:18: error: the deconstruction of T, whose type noncanonical.two has a user-defined equality, with determinism cc_multi is in a context that requires all solutions

:- pred sub_value(only_one::in, int::out) is det.
% CHECK: [[@LINE-1]]:9: error: the determinism declaration of sub_value(in, out) is not satisfied: it is declared det, but inferred multi
sub_value(T, N) :- T = one(N).
% CHECK: [[@LINE-1]]:24: because: the unification of T and one/1 can succeed more than once
% CHECK: [[@LINE-2]]:24: error: the deconstruction of T, whose type noncanonical.only_one has a user-defined equality, with determinism cc_multi is in a context that requires all solutions
