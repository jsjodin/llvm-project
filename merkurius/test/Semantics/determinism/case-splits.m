% Inside a case of a switch on a constant, mmc makes the case's test of
% the variable `true`, so that a disjunction after it is not split into
% tests of another variable, though a test after it still counts, and a
% test against a functor with arguments, which is kept, does not stop
% the split. A unification of a bound variable with a term whose
% arguments are bound tests fresh variables for the arguments, as in
% mmc, so the variable learns only the functor, not the arguments' insts.
%
% RUN: merkc -fc1 -fdebug-dump-determinism %s 2>%t.err | FileCheck %s
% RUN: not grep -E "error:|warning:" %t.err
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module case_splits.
:- interface.
:- type t ---> a ; b ; c(int) ; d(t, int).
:- type s ---> e ; f.
:- type r ---> g(int) ; h.
:- pred after_constant(t::in, s::in, int::out) is nondet.
% CHECK: pred case_splits.after_constant/3 mode 0: after_constant(in, in, out) is nondet
:- pred constant_last(t::in, s::in, int::out) is nondet.
% CHECK: pred case_splits.constant_last/3 mode 0: constant_last(in, in, out) is nondet
:- pred only_tests(t::in, s::in, int::out) is nondet.
% CHECK: pred case_splits.only_tests/3 mode 0: only_tests(in, in, out) is nondet
:- pred direct(t::in, s::in, int::out) is semidet.
% CHECK: pred case_splits.direct/3 mode 0: direct(in, in, out) is semidet
:- pred in_body(t::in, s::in, int::out) is semidet.
% CHECK: pred case_splits.in_body/3 mode 0: in_body(in, in, out) is semidet
:- pred with_arguments(t::in, r::in, int::out) is semidet.
% CHECK: pred case_splits.with_arguments/3 mode 0: with_arguments(in, in, out) is semidet
:- pred top_level(t::in, s::in, int::out) is semidet.
% CHECK: pred case_splits.top_level/3 mode 0: top_level(in, in, out) is semidet
:- pred bound_argument(t::in) is semidet.
% CHECK: pred case_splits.bound_argument/1 mode 0: bound_argument(in) is semidet
:- pred input_argument(t::in, int::in) is semidet.
% CHECK: pred case_splits.input_argument/2 mode 0: input_argument(in, in) is semidet
:- implementation.
:- import_module int.
after_constant(X, e, O) :- ( X = d(_, _), O = 2 ; X = c(K), O = K ).
after_constant(b, e, 1).
after_constant(b, f, 1).
constant_last(b, e, 1).
constant_last(X, e, O) :- ( X = d(_, _), O = 2 ; X = c(K), O = K ).
constant_last(b, f, 1).
only_tests(X, e, O) :- ( X = d(_, _) ; X = c(_) ), O = 2.
only_tests(b, e, 1).
only_tests(b, f, 1).
direct(X, e, O) :- X = d(_, _), O = 2.
direct(X, e, O) :- X = c(K), O = K.
direct(b, e, 1).
direct(b, f, 1).
in_body(X, S, O) :- S = e, ( X = d(_, _), O = 2 ; X = c(K), O = K ).
in_body(b, e, 1).
in_body(b, f, 1).
with_arguments(X, g(_), O) :- ( X = d(_, _), O = 2 ; X = c(K), O = K ).
with_arguments(b, g(_), 1).
with_arguments(b, h, 1).
top_level(d(_, _), e, 2).
top_level(X, e, O) :- ( X = c(K), O = K ; X = a, O = 0 ).
top_level(b, f, 1).
bound_argument(S) :- M = 1, S = c(M), S = c(0).
input_argument(S, M) :- M = 1, S = c(M), S = c(0).
