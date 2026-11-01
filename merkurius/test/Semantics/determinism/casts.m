% Determinism casts (reference manual, "Goals"), as mmc checks them:
% promise_equivalent_solutions and arbitrary succeed at most once where
% their goal can succeed more than once, and only the first solution of
% their goal is wanted, so it may make committed choice calls;
% promise_equivalent_solution_sets is as its goal. Each must list the
% variables its goal binds, those with free parts before it used outside
% it, as mmc's quantification finds them after mode analysis, none if the
% goal's end cannot be reached; an arbitrary scope must be in a
% promise_equivalent_solution_sets scope, outside lambda expressions,
% without its variables, and that scope must not be in another.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err | FileCheck %s
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: --implicit-check-not=because: %s < %t.err
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module casts.
:- interface.
:- type t ---> a ; b ; c(int).

:- pred multi_det(int::out) is det.
:- pred nondet_semidet(int::in, int::out) is semidet.
:- pred committed(int::out) is det.
:- pred no_solutions(int::out) is failure.
:- pred sets(int::out) is multi.
:- pred sets_det(int::out) is det.
:- pred arbitrary_semidet(int::in, int::out) is semidet.
:- pred no_arbitrary(int::out) is multi.
:- pred state(int::in, int::out) is det.
:- pred unlisted(int::out, int::out) is det.
:- pred unlisted_two(int::out, int::out, int::out) is det.
:- pred extra(int::in, int::out) is det.
:- pred extra_two(int::in, int::in, int::out) is det.
:- pred bound_before(int::in, int::out) is semidet.
:- pred local(int::out) is det.
:- pred not_nested(int::out) is det.
:- pred in_lambda(int::out) is det.
:- pred overlap(int::out) is det.
:- pred overlap_two(int::out, int::out) is det.
:- pred nested(int::out) is det.
% ERR-DAG: casts.m:[[@LINE+1]]:9: error: the determinism declaration of no_reasons(out) is not satisfied: it is declared failure, but inferred det
:- pred no_reasons(int::out) is failure.
% ERR-DAG: casts.m:[[@LINE+1]]:9: error: the determinism declaration of fail_reasons(in, out) is not satisfied: it is declared det, but inferred semidet
:- pred fail_reasons(int::in, int::out) is det.
% ERR-DAG: casts.m:[[@LINE+1]]:9: error: the determinism declaration of reasons(out) is not satisfied: it is declared failure, but inferred det
:- pred reasons(int::out) is failure.
:- pred requantified(t::in, int::in) is semidet.
:- pred not_requantified(int::in) is semidet.
:- pred failed(int::in) is semidet.
:- pred dead(int::in, int::out) is det.
:- pred reached(int::in, int::out) is semidet.
:- pred into_scopes(t::in, int::out) is det.
:- pred out_of_scopes(t::in, int::out) is nondet.
:- pred not_out_of_scopes(t::in, int::out) is nondet.

:- implementation.
:- import_module int.

:- pred m(int::out) is multi.
m(1).
m(2).
:- pred n(int::in, int::out) is nondet.
n(X, Y) :- ( Y = X ; X > 0, Y = 1 ).
:- pred cm(int::in, int::out) is cc_multi.
cm(X, X).
cm(X, X + 1).
:- pred fl(int::out) is failure.
fl(_) :- fail.
:- pred inc(int::in, int::out) is multi.
inc(X, X + 1).
inc(X, X + 2).

% CHECK-LABEL: pred casts.multi_det/1 mode 0: multi_det(out) is det, inferred det
% CHECK-NEXT:  promise_equivalent_solutions [X] [det]
% CHECK-NEXT:    call pred casts.m/1 mode 0: m(X) [multi]
multi_det(X) :- promise_equivalent_solutions [X] m(X).

% CHECK-LABEL: pred casts.nondet_semidet/2 mode 0: nondet_semidet(in, out) is semidet, inferred semidet
nondet_semidet(A, X) :- promise_equivalent_solutions [X] n(A, X).

% CHECK-LABEL: pred casts.committed/1 mode 0: committed(out) is det, inferred det
committed(X) :- promise_equivalent_solutions [X] ( m(Y), cm(Y, X) ).

% The goal cannot succeed, so it binds no variable.
% CHECK-LABEL: pred casts.no_solutions/1 mode 0: no_solutions(out) is failure, inferred failure
% ERR-DAG: casts.m:[[@LINE+1]]:20: error: the promise_equivalent_solutions goal lists an extra variable: X
no_solutions(X) :- promise_equivalent_solutions [X] fl(X).

% CHECK-LABEL: pred casts.sets/1 mode 0: sets(out) is multi, inferred multi
sets(X) :- promise_equivalent_solution_sets [X] ( arbitrary [Y] m(Y),
  ( X = Y ; X = Y + 1 ) ).

% CHECK-LABEL: pred casts.sets_det/1 mode 0: sets_det(out) is det, inferred det
sets_det(X) :- promise_equivalent_solution_sets [X] ( arbitrary [Y] m(Y),
  X = Y ).

% CHECK-LABEL: pred casts.arbitrary_semidet/2 mode 0: arbitrary_semidet(in, out) is semidet, inferred semidet
arbitrary_semidet(A, X) :- promise_equivalent_solution_sets [X]
  ( arbitrary [Y] n(A, Y), X = Y ).

% Without an arbitrary scope, which the manual says it should have, mmc
% casts nothing, and reports nothing.
% CHECK-LABEL: pred casts.no_arbitrary/1 mode 0: no_arbitrary(out) is multi, inferred multi
no_arbitrary(X) :- promise_equivalent_solution_sets [X] m(X).

% !:S is the variable bound.
% CHECK-LABEL: pred casts.state/2 mode 0: state(in, out) is det, inferred det
% CHECK-NEXT:  promise_equivalent_solutions [STATE_VARIABLE_S_1] [det]
state(!S) :- promise_equivalent_solutions [!:S] inc(!S).

% ERR-DAG: casts.m:[[@LINE+1]]:19: error: the promise_equivalent_solutions goal binds a variable that is not listed: Y
unlisted(X, Y) :- promise_equivalent_solutions [X] ( m(X), m(Y) ).

% In the order of the variables.
% ERR-DAG: casts.m:[[@LINE+1]]:26: error: the promise_equivalent_solutions goal binds some variables that are not listed: Z, Y and X
unlisted_two(Z, Y, X) :- promise_equivalent_solutions [] ( m(X), m(Y), m(Z) ).

% ERR-DAG: casts.m:[[@LINE+1]]:16: error: the promise_equivalent_solutions goal lists an extra variable: A
extra(A, X) :- promise_equivalent_solutions [X, A] m(X).

% ERR-DAG: casts.m:[[@LINE+1]]:23: error: the promise_equivalent_solutions goal lists some extra variables: A and B
extra_two(A, B, X) :- promise_equivalent_solutions [B, X, A] m(X).

% X is bound before.
% ERR-DAG: casts.m:[[@LINE+1]]:30: error: the promise_equivalent_solutions goal lists an extra variable: X
bound_before(A, X) :- X = A, promise_equivalent_solutions [X] ( m(Y), Y = X ).

% Z is local to the goal.
% ERR-DAG: casts.m:[[@LINE+1]]:13: error: the promise_equivalent_solutions goal lists an extra variable: Z
local(X) :- promise_equivalent_solutions [X, Z] ( m(Z), X = Z ).

% mmc casts it all the same.
% CHECK-LABEL: pred casts.not_nested/1 mode 0: not_nested(out) is det, inferred det
% ERR-DAG: casts.m:[[@LINE+1]]:18: error: this arbitrary scope is not nested inside a promise_equivalent_solution_sets scope
not_nested(X) :- arbitrary [X] m(X).

% ERR-DAG: casts.m:[[@LINE+2]]:31: error: this arbitrary scope is not nested inside a promise_equivalent_solution_sets scope
in_lambda(X) :- promise_equivalent_solution_sets [X] ( arbitrary [Y] m(Y),
  P = (pred(Z::out) is det :- arbitrary [Z] m(Z)), P(W), X = Y + W ).

% ERR-DAG: casts.m:[[@LINE+2]]:54: error: this arbitrary scope and the promise_equivalent_solution_sets scope it is nested inside overlap on the variable X
% ERR-DAG: casts.m:[[@LINE+1]]:15: this is the outer promise_equivalent_solution_sets scope
overlap(X) :- promise_equivalent_solution_sets [X] ( arbitrary [X] m(X) ).

% ERR-DAG: casts.m:[[@LINE+2]]:64: error: this arbitrary scope and the promise_equivalent_solution_sets scope it is nested inside overlap on the following variables: X and Y
% ERR-DAG: casts.m:[[@LINE+1]]:22: this is the outer promise_equivalent_solution_sets scope
overlap_two(X, Y) :- promise_equivalent_solution_sets [X, Y] ( arbitrary [Y, X] ( m(X), m(Y) ) ).

% ERR-DAG: casts.m:[[@LINE+3]]:3: error: this promise_equivalent_solution_sets scope is nested inside another
% ERR-DAG: casts.m:[[@LINE+1]]:14: this is the outer promise_equivalent_solution_sets scope
nested(X) :- promise_equivalent_solution_sets [X] ( arbitrary [Y] m(Y),
  promise_equivalent_solution_sets [X] ( arbitrary [Z] m(Z),
  X = Y + Z ) ).

% Inside a cast whose goal can succeed more than once, as in mmc, the
% reasons are only the goals that make it fail, and here there are none;
% inside one whose goal cannot, they are as elsewhere.
no_reasons(X) :- promise_equivalent_solutions [X] m(X).

% ERR-DAG: casts.m:[[@LINE+1]]:56: because: the call to n(in, out) can fail
fail_reasons(A, X) :- promise_equivalent_solutions [X] n(A, X).

% ERR-DAG: casts.m:[[@LINE+1]]:54: because: the unification of X and 1 can succeed
reasons(X) :- promise_equivalent_solutions [X] ( X = 1 ).

% R = 2 cannot succeed after R is 0 or 1. When mmc checks the modes
% again, after W = c(_) is moved out of the disjunction, it is fail, R is
% no longer used outside the cast, and the cast binds nothing; without
% that, R is still bound.
% ERR-DAG: casts.m:[[@LINE+1]]:29: error: the promise_equivalent_solutions goal lists an extra variable: R
requantified(W, H) :- not ( promise_equivalent_solutions [R] ( W = c(_), R = 0 ;
  W = c(H), R = 1 ), R = 2 ).
not_requantified(A) :- not ( promise_equivalent_solutions [Y]
  ( A = 0, Y = 1 ; A = 1, Y = 2 ), Y = 3 ).

% Before that, the variables of the unification that cannot succeed are
% still used outside: Y is bound, and the disjunction must find all of Y's
% values but the first.
% CHECK-LABEL: pred casts.failed/1 mode 0: failed(in) is semidet, inferred det
% CHECK:       disjunction [cc_multi]
failed(A) :- not ( promise_equivalent_solutions [Y] ( Y = 1 ; A = 0, Y = 2 ),
  Y = 3 ).

% mmc's mode analysis deletes R1 = M, since R1 is not used after it, so M
% is not used outside the arbitrary scope.
% ERR-DAG: casts.m:[[@LINE+2]]:5: error: the arbitrary goal lists an extra variable: M
dead(W, R) :- promise_equivalent_solution_sets [R] (
  ( arbitrary [M] m(M), W = 1, R1 = M ; R1 = 3 ), R = 0 ).

% The end of the goal is reached in the first disjunct, though it cannot
% succeed, so X is bound; in the second X = c(_) cannot succeed.
reached(A, Y) :- B = a, promise_equivalent_solutions [Y] ( not A = A, Y = 0 ;
  B = c(_), Y = 1 ).

% A test of X inside a scope makes a switch on X, as in mmc.
% CHECK-LABEL: pred casts.into_scopes/2 mode 0: into_scopes(in, out) is det, inferred det
% CHECK-NEXT:  switch on X, cannot fail [det]
into_scopes(X, Y) :- ( promise_pure ( X = a, Y = 1 ) ; X = b, Y = 2 ;
  promise_equivalent_solutions [Y] ( X = c(_), Y = 3 ) ).

% And one in every disjunct is moved out of the disjunction.
% CHECK-LABEL: pred casts.out_of_scopes/2 mode 0: out_of_scopes(in, out) is nondet, inferred nondet
% CHECK-NEXT:  deconstruct X = casts.c/1
out_of_scopes(X, Y) :- ( promise_equivalent_solutions [Y] ( X = c(K), Y = K ) ;
  X = c(_), Y = 0 ).

% But not past a call in a scope.
% CHECK-LABEL: pred casts.not_out_of_scopes/2 mode 0: not_out_of_scopes(in, out) is nondet, inferred nondet
% CHECK-NEXT:  disjunction [nondet]
not_out_of_scopes(X, Y) :- ( promise_equivalent_solutions [Z] m(Z), X = c(K),
  Y = K + Z ; X = c(_), Y = 0 ).
