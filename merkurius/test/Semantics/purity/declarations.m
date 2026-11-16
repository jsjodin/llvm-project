% A predicate's or function's purity is the least of its clauses' goals'
% (reference manual, "Promising that a predicate is pure"): declared purer
% than that is an error, and less pure a warning. A promise_pure or
% promise_semipure pragma, or scope, stops the impurity, and a pragma must
% promise the declared purity; one that is not needed is a warning. A
% predicate without a declaration is pure, as mmc infers no purity, and a
% trace goal's impurity does not leak.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module declarations.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int.

main(0).
:- impure pred ip(int::in) is det.
:- pragma foreign_proc("C", ip(X::in), [will_not_call_mercury], "(void) X;").
:- semipure pred sp(int::out) is det.
:- pragma foreign_proc("C", sp(X::out), [will_not_call_mercury, promise_semipure], "X = 0;").
:- impure func ifn(int) = int.
:- pragma foreign_proc("C", ifn(X::in) = (Y::out), [will_not_call_mercury], "Y = X;").
:- pred pp(int::in) is det.
pp(_).

% Declared as pure as the clauses.
:- impure pred impure_ok(int::out) is det.
impure_ok(X) :- semipure sp(X), impure ip(X).
:- semipure pred semipure_ok(int::out) is det.
semipure_ok(X) :- semipure sp(X).
:- impure func impure_function = int.
impure_function = X :- impure X = ifn(1).

% Declared purer than the clauses.
:- pred pure_but_impure is det.
% CHECK: [[@LINE-1]]:9: error: pred declarations.pure_but_impure/0 is impure, so it must be declared `impure' or promised pure
pure_but_impure :- impure ip(1).
:- semipure pred semipure_but_impure is det.
% CHECK: [[@LINE-1]]:18: error: pred declarations.semipure_but_impure/0 is impure, so it must be declared `impure' or promised semipure
semipure_but_impure :- impure ip(1).
:- pred pure_but_semipure(int::out) is det.
% CHECK: [[@LINE-1]]:9: error: pred declarations.pure_but_semipure/1 is semipure, so it must be declared `semipure' or promised pure
pure_but_semipure(X) :- semipure sp(X).
:- func function_but_impure = int.
% CHECK: [[@LINE-1]]:9: error: func declarations.function_but_impure/0 is impure, so it must be declared `impure' or promised pure
function_but_impure = X :- impure X = ifn(1).
% The least of the clauses'.
:- pred clauses(int::in) is semidet.
% CHECK: [[@LINE-1]]:9: error: pred declarations.clauses/1 is impure, so it must be declared `impure' or promised pure
clauses(1) :- pp(1).
clauses(2) :- impure ip(1).
% Without a declaration, and so pure.
undeclared :- impure ip(1).
% CHECK: [[@LINE-1]]:1: error: pred declarations.undeclared/0 is impure, so it must be declared `impure' or promised pure
% Unannotated, the call still counts.
:- pred unannotated is det.
% CHECK: [[@LINE-1]]:9: error: pred declarations.unannotated/0 is impure, so it must be declared `impure' or promised pure
unannotated :- ip(1).
% CHECK: [[@LINE-1]]:16: error: the call of impure pred declarations.ip/1 must be annotated `impure'

% Declared less pure than the clauses.
:- impure pred impure_but_pure is det.
% CHECK: [[@LINE-1]]:16: warning: pred declarations.impure_but_pure/0 is declared impure, but it is pure
impure_but_pure :- pp(1).
:- impure pred impure_but_semipure(int::out) is det.
% CHECK: [[@LINE-1]]:16: warning: pred declarations.impure_but_semipure/1 is declared impure, but it is semipure
impure_but_semipure(X) :- semipure sp(X).
:- semipure pred semipure_but_pure is det.
% CHECK: [[@LINE-1]]:18: warning: pred declarations.semipure_but_pure/0 is declared semipure, but it is pure
semipure_but_pure :- pp(1).

% Promises.
:- pred promised is det.
:- pragma promise_pure(promised/0).
promised :- impure ip(1).
:- func promised_function(int) = int.
:- pragma promise_pure(promised_function/1).
promised_function(X) = X :- impure ip(1).
:- semipure pred promised_semipure(int::out) is det.
:- pragma promise_semipure(promised_semipure/1).
promised_semipure(X) :- impure ip(1), semipure sp(X).
:- pred unneeded_promise is det.
% CHECK: [[@LINE-1]]:9: warning: the `promise_pure' pragma is unnecessary, since pred declarations.unneeded_promise/0 is pure
:- pragma promise_pure(unneeded_promise/0).
unneeded_promise :- pp(1).
:- impure pred impure_promised_pure is det.
% CHECK: [[@LINE-1]]:16: error: pred declarations.impure_promised_pure/0 is declared impure, but promised pure
:- pragma promise_pure(impure_promised_pure/0).
impure_promised_pure :- impure ip(1).
:- semipure pred semipure_promised_pure(int::out) is det.
% CHECK: [[@LINE-1]]:18: error: pred declarations.semipure_promised_pure/1 is declared semipure, but promised pure
:- pragma promise_pure(semipure_promised_pure/1).
semipure_promised_pure(X) :- semipure sp(X).
:- pragma promise_pure(undeclared_promised/0).
undeclared_promised :- impure ip(1).

% Promise scopes.
:- pred scope is det.
scope :- promise_pure ( impure ip(1) ).
:- pred semipure_scope is det.
% CHECK: [[@LINE-1]]:9: error: pred declarations.semipure_scope/0 is semipure, so it must be declared `semipure' or promised pure
semipure_scope :- promise_semipure ( impure ip(1) ).
:- semipure pred semipure_scope_ok is det.
semipure_scope_ok :- promise_semipure ( pp(1) ).
:- pred impure_scope is det.
% CHECK: [[@LINE-1]]:9: error: pred declarations.impure_scope/0 is impure, so it must be declared `impure' or promised pure
impure_scope :- promise_impure ( pp(1) ).
:- pred nested_scopes is det.
nested_scopes :- promise_pure ( impure ip(1), promise_semipure ( impure ip(2) ) ).
:- pred unneeded_scope is det.
unneeded_scope :- promise_pure ( pp(1) ).

% A trace goal.
:- pred traced is det.
traced :- trace [compile_time(flag("x"))] ( impure ip(1) ).
