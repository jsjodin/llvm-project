% Switch detection and common subexpression elimination (Henderson,
% Somogyi and Conway, "Determinism analysis in the Mercury compiler", 1996,
% sections 3 and 4; reference manual, "Switches"), as mmc does them: a
% disjunction whose disjuncts test a variable against different functors,
% through aliases, after other unifications, in `some` scopes, or split at
% a disjunction, is a switch, which can fail if the variable's inst allows
% a functor without a case; a partial switch has the other disjuncts after
% it; the clauses of a predicate are a disjunction on the head variables;
% a test that every disjunct makes with the same functor is moved out of
% the disjunction, and its arguments may then be switched on; and mmc's
% choices between variables.
%
% RUN: merkc -fc1 -fdebug-dump-determinism %s | FileCheck %s
% RUN: %if mmc %{ %compare-switches %s %}

:- module switches.
:- interface.
:- import_module list.

:- type t ---> a ; b ; c(int) ; d(t, int).
:- type yn ---> y ; n.
:- type w ---> w(yn, int).
:- inst ab == bound(a ; b).

:- pred full(t::in, int::out) is det.
:- pred inst_covers(t::in(ab), int::out) is det.
:- pred aliased(t::in, int::out) is semidet.
:- pred partial(t::in, int::out) is nondet.
:- pred split(t::in, int::out) is multi.
:- pred clauses(list(int)::in, int::out) is det.
:- pred integers(int::in, int::out) is semidet.
:- pred common(w::in, int::out) is det.
:- pred cannot_fail_first(t::in, yn::in, int::out) is semidet.
:- pred one_functor(t::in, t::in, int::out) is nondet.
:- pred integer_first(t::in, int::in, int::out) is nondet.
:- pred in_lambda(t::in, int::out) is nondet.
:- pred merged(t::in, int::in, int::out) is nondet.
:- pred alias_kept(t::in, int::out) is nondet.
:- pred fail_clause(t::in, t::in(ab), int::out) is nondet.
:- pred scope_alias(list(int)::in, int::out) is nondet.
:- pred deleted is det.
:- pred alias_common(int::in, int::out) is semidet.
:- pred one_left(t::in, t::in, int::out) is semidet.

:- implementation.
:- import_module int.

:- pred q(int::in, int::out) is det.
q(X, X).

full(X, Y) :- ( X = a, Y = 1 ; X = b, Y = 2 ; X = c(Y) ; X = d(_, Y) ).

% CHECK-LABEL: pred switches.full/2 mode 0: full(in, out) is det
% CHECK-NEXT:  switch on X, cannot fail
% CHECK-NEXT:    case switches.a/0
% CHECK-NEXT:      deconstruct X = switches.a/0, switched on
% CHECK-NEXT:      construct Y = 1
% CHECK-NEXT:    case switches.b/0
% CHECK-NEXT:      deconstruct X = switches.b/0, switched on
% CHECK-NEXT:      construct Y = 2
% CHECK-NEXT:    case switches.c/1
% CHECK-NEXT:      deconstruct X = switches.c/1(Y), switched on
% CHECK-NEXT:    case switches.d/2
% CHECK-NEXT:      deconstruct X = switches.d/2(V_{{[0-9]+}}, Y), switched on

% Only a and b are allowed, so these cases cover X.
inst_covers(X, Y) :- ( X = a, Y = 1 ; X = b, Y = 2 ).

% CHECK-LABEL: pred switches.inst_covers/2 mode 0:
% CHECK-NEXT:  switch on X, cannot fail

% Tests through variables unified with X, after a construction, and in a
% `some` scope.
aliased(X, Y) :- ( Z = X, M = 1, Z = a, Y = M ; some [K] X = c(K), Y = 2 ).

% CHECK-LABEL: pred switches.aliased/2 mode 0:
% CHECK-NEXT:  switch on X, can fail
% CHECK-NEXT:    case switches.a/0
% CHECK-NEXT:      assign Z := X
% CHECK-NEXT:      construct M = 1
% CHECK-NEXT:      deconstruct Z = switches.a/0, switched on
% CHECK-NEXT:      assign Y := M
% CHECK-NEXT:    case switches.c/1
% CHECK-NEXT:      some
% CHECK-NEXT:        deconstruct X = switches.c/1(K), switched on
% CHECK-NEXT:      construct Y = 2

% A call before the test leaves the disjunct out of the switch, which is
% followed by it.
partial(X, Y) :- ( X = a, Y = 1 ; q(1, M), X = b, Y = M ; X = b, Y = 3 ).

% CHECK-LABEL: pred switches.partial/2 mode 0:
% CHECK-NEXT:  disjunction
% CHECK-NEXT:    branch
% CHECK-NEXT:      switch on X, can fail
% CHECK-NEXT:        case switches.a/0
% CHECK:             case switches.b/0
% CHECK:           branch
% CHECK-NEXT:      construct V_{{[0-9]+}} = 1
% CHECK-NEXT:      call pred switches.q/2 mode 0: q(V_{{[0-9]+}}, M)

% Each branch of the inner disjunction, with the goal after it, tests X,
% so the disjunct is split into one for each; those parts come after the
% disjuncts that test X.
split(X, Y) :- ( ( X = a, Y0 = 1 ; X = b, Y0 = 2 ), Y = Y0 ; X = a, Y = 3 ;
  X = c(_), Y = 4 ; X = d(_, _), Y = 5 ).

% CHECK-LABEL: pred switches.split/2 mode 0:
% CHECK-NEXT:  switch on X, cannot fail
% CHECK-NEXT:    case switches.a/0
% CHECK-NEXT:      disjunction
% CHECK-NEXT:        branch
% CHECK-NEXT:          deconstruct X = switches.a/0, switched on
% CHECK-NEXT:          construct Y = 3
% CHECK-NEXT:        branch
% CHECK-NEXT:          deconstruct X = switches.a/0, switched on
% CHECK-NEXT:          construct Y0 = 1
% CHECK-NEXT:          assign Y := Y0
% CHECK-NEXT:    case switches.b/0
% CHECK-NEXT:      deconstruct X = switches.b/0, switched on
% CHECK-NEXT:      construct Y0 = 2
% CHECK-NEXT:      assign Y := Y0

% The clauses test the head variable, and [_] and [_, _ | _] both have
% [|], which is moved out of the disjunction, so that its tail is
% switched on. As in mmc, the test against [] that the switch makes is
% then deleted.
clauses([], 0).
clauses([_], 1).
clauses([_, _ | _], 2).

% CHECK-LABEL: pred switches.clauses/2 mode 0:
% CHECK-NEXT:  switch on HeadVar__1, cannot fail
% CHECK-NEXT:    case list.[]/0
% CHECK-NEXT:      construct HeadVar__2 = 0
% CHECK-NEXT:    case list.[|]/2
% CHECK-NEXT:      deconstruct HeadVar__1 = list.[|]/2(V_[[H:[0-9]+]], V_[[T:[0-9]+]])
% CHECK-NEXT:      switch on V_[[T]], cannot fail
% CHECK-NEXT:        case list.[]/0
% CHECK-NEXT:          assign V_[[A:[0-9]+]] := V_[[T]]
% CHECK-NEXT:          deconstruct V_[[A]] = list.[]/0, switched on
% CHECK-NEXT:          construct HeadVar__2 = 1
% CHECK-NEXT:        case list.[|]/2

% The cases of a switch on an integer are in order, and it can fail.
integers(N, Y) :- ( N = 3, Y = 0 ; N = 1, Y = 1 ; N = 2, Y = 2 ).

% CHECK-LABEL: pred switches.integers/2 mode 0:
% CHECK-NEXT:  switch on N, can fail
% CHECK-NEXT:    case 1
% CHECK:         case 2
% CHECK:         case 3

% w has one functor, so W is not switched on; the deconstruction is moved
% out, and its first argument switched on.
common(W, Y) :- ( W = w(y, Y) ; W = w(n, K), Y = K + 1 ).

% CHECK-LABEL: pred switches.common/2 mode 0:
% CHECK-NEXT:  deconstruct W = switches.w/2(V_[[A:[0-9]+]], V_{{[0-9]+}})
% CHECK-NEXT:  switch on V_[[A]], cannot fail
% CHECK-NEXT:    case switches.n/0
% CHECK:         case switches.y/0

% Of two variables that every disjunct tests, the switch that cannot fail
% is chosen.
cannot_fail_first(X, Y, Z) :- ( X = a, Y = y, Z = 1 ; X = b, Y = n, Z = 2 ).

% CHECK-LABEL: pred switches.cannot_fail_first/3 mode 0:
% CHECK-NEXT:  switch on Y, cannot fail

% Every disjunct tests X against b, so mmc does not switch on W, though it
% could, and moves X = b out of the disjunction instead; W is then
% switched on.
one_functor(W, X, Y) :- ( W = a, X = b, Y = 1 ; W = c(_), X = b, Y = 2 ).

% CHECK-LABEL: pred switches.one_functor/3 mode 0:
% CHECK-NEXT:  deconstruct X = switches.b/0
% CHECK-NEXT:  switch on W, can fail

% A switch on an integer is chosen over one on a type with constructors
% that can fail.
integer_first(X, N, Y) :- ( X = a, N = 3, Y = 0 ; X = b, N = 0, Y = 3 ).

% CHECK-LABEL: pred switches.integer_first/3 mode 0:
% CHECK-NEXT:  switch on N, can fail

% Switches in a lambda expression's body.
in_lambda(X, Y) :-
  P = (pred(Z::out) is nondet :- ( X = a, Z = 1 ; X = b, Z = 2 )),
  P(Y).

% CHECK-LABEL: pred switches.in_lambda/2 mode 0:
% CHECK-NEXT:  construct P = lambda
% CHECK-NEXT:    switch on X, can fail

% A disjunct that starts with a disjunction of tests of X alone is in the
% case of each functor without them, and the cases that have only it are
% one.
merged(X, H, Y) :- ( ( X = b ; X = c(_) ; X = d(_, _) ), Y = H ; X = a, Y = 1 ).

% CHECK-LABEL: pred switches.merged/3 mode 0:
% CHECK-NEXT:  switch on X, cannot fail
% CHECK-NEXT:    case switches.a/0
% CHECK-NEXT:      deconstruct X = switches.a/0, switched on
% CHECK-NEXT:      construct Y = 1
% CHECK-NEXT:    case switches.b/0 or switches.c/1 or switches.d/2
% CHECK-NEXT:      assign Y := H

% In the case b of X, the switch on X has no case that can be reached, and
% is fail; the test of X through Z is kept, as in mmc, though the one of
% X is deleted.
alias_kept(X, Y) :-
  ( X = b, ( Z = X, Z = b, ( X = a ; X = c(_) ), Y = 0 ; Y = 1 ) ; X = a, Y = 2 ).

% CHECK-LABEL: pred switches.alias_kept/2 mode 0:
% CHECK:         case switches.b/0
% CHECK-NEXT:      deconstruct X = switches.b/0, switched on
% CHECK-NEXT:      disjunction
% CHECK-NEXT:        branch
% CHECK-NEXT:          assign Z := X
% CHECK-NEXT:          deconstruct Z = switches.b/0
% CHECK-NEXT:          fail
% CHECK-NEXT:        branch
% CHECK-NEXT:          construct Y = 1

% The second argument cannot be c(_), so each clause fails, after the
% head unifications written before it, which are run first.
fail_clause(b, c(K), K).
fail_clause(X, c(K), Y) :- ( X = c(Y) ; X = a, Y = K ).

% CHECK-LABEL: pred switches.fail_clause/3 mode 0:
% CHECK-NEXT:  disjunction
% CHECK-NEXT:    branch
% CHECK-NEXT:      deconstruct HeadVar__1 = switches.b/0
% CHECK-NEXT:      fail
% CHECK-NEXT:    branch
% CHECK-NEXT:      assign X := HeadVar__1
% CHECK-NEXT:      fail

% T is unified with the tail of L in a `some` scope, and the disjunction
% after it is not split on it, as in mmc, so the first disjunct is left
% after the switch on the tail.
scope_alias(L, Y) :-
  ( some [H] (L = [H | T]), ( T = [_ | _], Y = 1 ; T = [], Y = 2 ) ;
    L = [_], Y = 3 ; L = [_, _ | _], Y = 4 ).

% CHECK-LABEL: pred switches.scope_alias/2 mode 0:
% CHECK-NEXT:  deconstruct L = list.[|]/2(V_{{[0-9]+}}, V_[[T:[0-9]+]])
% CHECK-NEXT:  disjunction
% CHECK-NEXT:    branch
% CHECK-NEXT:      switch on V_[[T]], cannot fail
% CHECK:         branch
% CHECK-NEXT:      some
% CHECK-NEXT:        assign T := V_[[T]]
% CHECK-NEXT:      switch on T, cannot fail

% The body is a construction of a variable not used, which mmc deletes.
deleted :- _X = 2.

% CHECK-LABEL: pred switches.deleted/0 mode 0:
% CHECK-NEXT:  {{[0-9]+}}:{{[0-9]+}} true

% In the case 0 of the switch on X, the two disjuncts test X through
% Z and V, which mmc keeps, and moves out of the disjunction as a test of
% X. With X known then, X = 2 cannot succeed, and the disjunct is fail and
% deleted.
alias_common(X, Y) :- ( Z = X, Z = 0, Y = 1 ; V = X, V = 0, X = 2, Y = 2 ;
  X = 3, Y = 3 ).

% CHECK-LABEL: pred switches.alias_common/2 mode 0:
% CHECK-NEXT:  switch on X, can fail [semidet]
% CHECK-NEXT:    case 0
% CHECK-NEXT:      deconstruct X = 0 [det]
% CHECK-NEXT:      construct Y = 1 [det]
% CHECK-NEXT:    case 3
% CHECK-NEXT:      construct Y = 3 [det]

% After the deconstruction of HeadVar__1 is moved out, the branch that
% cannot succeed is deleted, and the disjunction left with one branch is
% part of the first clause's conjunction, which tests HeadVar__2 then.
one_left(c(0), W, Y) :- ( W = d(_, _), Y = 0 ; M = 2, M = 1, Y = M ).
one_left(c(_), c(K), K).

% CHECK-LABEL: pred switches.one_left/3 mode 0:
% CHECK-NEXT:  deconstruct HeadVar__1 = switches.c/1
% CHECK-NEXT:  switch on HeadVar__2, can fail [semidet]
% CHECK-NEXT:    case switches.c/1
% CHECK:         case switches.d/2
