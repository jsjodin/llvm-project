% The determinism of goals (reference manual, "Determinism checking and
% inference"; Henderson, Somogyi and Conway, "Determinism analysis in the
% Mercury compiler", 1996, section 5.1), as mmc infers it: of calls, by
% their callees' declarations, of higher-order calls by their insts, and of
% function applications; of unifications, a deconstruction being det if
% its variable's inst, a switch or a deconstruction before, through an
% assignment, gives its functor; of conjunctions, disjunctions, switches,
% if-then-elses, the else part not counting if the condition cannot fail,
% negations and scopes; and a goal with more than one solution but no
% outputs, but not an impure one, has at most one, the innermost such
% goal. A declaration that is looser than the inferred determinism has a
% warning. A procedure that calls one without a declared determinism is
% checked by a later step; committed choice is in committed.m, determinism
% casts in casts.m and requirements in requires.m.
%
% RUN: merkc -fc1 -fdebug-dump-determinism %s 2>%t.err | FileCheck %s
% RUN: FileCheck --check-prefix=WARN --implicit-check-not=warning: --implicit-check-not=error: %s < %t.err
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module goals.
:- interface.
:- import_module list.

:- type t ---> a ; b ; c(int).
:- type unit ---> unit.

:- pred calls(int::in, int::out) is nondet.
:- pred higher_order(pred(int)::in(pred(out) is semidet), int::out) is semidet.
:- pred apply(int::in, int::out) is semidet.
:- pred deconstruct(t::in, int::out) is semidet.
:- pred known(t::in, int::out) is det.
:- pred aliased(t::in, int::out) is det.
:- pred unit_test(unit::in, unit::in) is semidet.
:- pred disjunction(int::out) is multi.
:- pred one_disjunct(int::out) is multi.
:- pred if_then_else(int::in, int::out) is nondet.
:- pred cannot_fail(int::out) is multi.
:- pred negation(int::in) is semidet.
:- pred no_outputs(int::in) is semidet.
:- pred nested_no_outputs(int::in, int::out) is det.
:- impure pred impure_no_outputs(int::in) is multi.
:- pred some_scope(list(int)::in) is semidet.
:- pred erroneous(int::in, int::out) is det.
:- pred lambda(pred(int)::out(pred(out) is multi)) is det.
:- pred too_loose(int::out) is semidet.
:- func default(int) = int.
:- func explicit(int) = int.
:- mode explicit(in) = out is multi.
:- pred inferred_callee(int::out) is det.

:- implementation.
:- import_module int, require.

:- pred sd(int::in, int::out) is semidet.
sd(X, X) :- X > 0.
:- pred m(int::out) is multi.
m(1).
m(2).
:- func f(int) = int is semidet.
f(X) = X :- X > 0.
:- impure pred touch(int::in, int::out) is multi.
% WARN-DAG: warning: pred goals.touch/2 is declared impure, but it is pure
touch(X, Y) :- ( Y = X ; Y = X + 1 ).
:- pred undeclared(int::out).
undeclared(1).

% CHECK-LABEL: pred goals.calls/2 mode 0: calls(in, out) is nondet, inferred nondet
% CHECK: call pred goals.sd/2 mode 0: sd(X, Z) [semidet]
% CHECK: call pred goals.m/1 mode 0: m(W) [multi]
calls(X, Y) :- sd(X, Z), m(W), Y = Z + W.

% CHECK-LABEL: higher_order(in((pred(out) is semidet)), out) is semidet, inferred semidet
% CHECK: higher-order call: P(Y) [semidet]
higher_order(P, Y) :- P(Y).

% CHECK-LABEL: apply(in, out) is semidet, inferred semidet
% CHECK: call func goals.f/1 mode 0: Y = f(X) [semidet]
apply(X, Y) :- Y = f(X).

% CHECK-LABEL: deconstruct(in, out) is semidet, inferred semidet
% CHECK: deconstruct X = goals.c/1(Y) [semidet]
deconstruct(X, Y) :- X = c(Y).

% CHECK-LABEL: known(in, out) is det, inferred det
% CHECK: deconstruct X = goals.c/1(Z), switched on [det]
% CHECK: deconstruct X = goals.c/1(W) [det]
known(X, Y) :- ( X = a, Y = 1 ; X = c(Z), X = c(W), Y = Z + W ; X = b, Y = 2 ).

% CHECK-LABEL: aliased(in, out) is det, inferred det
% CHECK: assign Z := X [det]
% CHECK: deconstruct Z = goals.c/1(Y) [det]
aliased(X, Y) :- ( X = c(_), Z = X, Z = c(Y) ; X = a, Y = 0 ; X = b, Y = 1 ).

% A unification of two variables can fail, even of a type with one value.
% CHECK-LABEL: unit_test(in, in) is semidet, inferred semidet
% CHECK: test X = Y [semidet]
unit_test(X, Y) :- X = Y.

% CHECK-LABEL: disjunction(out) is multi, inferred multi
% CHECK: disjunction [multi]
disjunction(Y) :- ( Y = 1 ; Y = 2 ).

% A disjunct that fails is left out.
% CHECK-LABEL: one_disjunct(out) is multi, inferred multi
% CHECK-NEXT: call pred goals.m/1 mode 0: m(Y) [multi]
one_disjunct(Y) :- ( m(Y) ; fail ).

% CHECK-LABEL: if_then_else(in, out) is nondet, inferred nondet
% CHECK: if-then-else [nondet]
if_then_else(X, Y) :- ( if X > 0 then m(Y) else sd(X, Y) ).

% CHECK-LABEL: cannot_fail(out) is multi, inferred multi
% CHECK: if-then-else [multi]
cannot_fail(Y) :- ( if m(Z) then Y = Z else fail ).

% CHECK-LABEL: negation(in) is semidet, inferred semidet
% CHECK: not [semidet]
negation(X) :- not ( m(Y), Y = X ).

% CHECK-LABEL: no_outputs(in) is semidet, inferred semidet
% CHECK: call pred goals.m/1 mode 0: m(Y) [multi]
no_outputs(X) :- m(Y), Y > X.

% CHECK-LABEL: nested_no_outputs(in, out) is det, inferred det
% CHECK: disjunction [det]
% CHECK: disjunction [semidet]
nested_no_outputs(X, Y) :- ( X > 0, ( X < 5 ; X > 9 ) ; true ), Y = X.

% CHECK-LABEL: impure_no_outputs(in) is multi, inferred multi
impure_no_outputs(X) :- impure touch(X, _).

% CHECK-LABEL: some_scope(in) is semidet, inferred semidet
% CHECK: some [semidet]
some_scope(L) :- some [X] ( list.member(X, L), X > 3 ).

% After a call that cannot succeed, nothing is reached.
% CHECK-LABEL: erroneous(in, out) is det, inferred det
% CHECK: if-then-else [det]
% CHECK: call pred require.error/1 mode 0: error(V_{{[0-9]+}}) [erroneous]
erroneous(X, Y) :- ( if X > 0 then Y = X else error("no") ).

% CHECK-LABEL: lambda(out((pred(out) is multi))) is det, inferred det
% CHECK: construct P = lambda() [det]
% CHECK: call pred goals.m/1 mode 0: m(Y) [multi]
lambda(P) :- P = (pred(Y::out) is multi :- m(Y)).

% WARN-DAG: warning: the determinism declaration of too_loose(out) could be tighter: it is declared semidet, but inferred det
too_loose(1).

% A function's default mode is det.
% CHECK-LABEL: (default) func goals.default/1 mode 0: default(in) = out is det, inferred det
default(X) = X.

% WARN-DAG: warning: the determinism declaration of explicit(in) = out could be tighter: it is declared multi, but inferred det
explicit(X) = X.

% Determinism inference is a later step.
% CHECK-LABEL: inferred_callee(out) is det{{$}}
inferred_callee(Y) :- undeclared(Y).
