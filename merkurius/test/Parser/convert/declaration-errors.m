% Malformed declarations: a qualified name whose last part is not a name, a
% type attribute that is not one or is given twice, constructors that are
% not ones, an equivalence type with attributes, a type class without
% parameters, a function declaration without its result, a mode declaration
% with a type, a declaration that is not one, a type that is not one, a mode
% that is not one, and an instance method that is not a clause. mmc reports
% an error on each of these lines too: for the mode declaration with a type,
% that its predicate ':'/2 is undeclared.
%
% RUN: not merkc -fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module declaration_errors.
:- implementation.
:- import_module int, list.
:- type 'X'(A).3 ---> c.
% CHECK: declaration-errors.m:[[@LINE-1]]:16: error: expected a name, but found '3'
:- type t1 ---> c1(int) where frob is x.
% CHECK: declaration-errors.m:[[@LINE-1]]:31: error: expected 'equality is', 'comparison is', 'direct_arg is' or 'type_is_abstract_noncanonical', but found 'frob is x'
:- type t2 ---> c2(int) where equality is eq2, equality is eq2.
% CHECK: declaration-errors.m:[[@LINE-1]]:48: error: the type attribute 'equality' is given more than once
:- pred eq2(t2::in, t2::in) is semidet.
eq2(_, _).
:- type t3 ---> 42.
% CHECK: declaration-errors.m:[[@LINE-1]]:17: error: expected a constructor, but found '42'
:- type t4 ---> c4 ; 42.
% CHECK: declaration-errors.m:[[@LINE-1]]:22: error: expected a constructor, but found '42'
:- type t5 == int where equality is eq5.
% CHECK: declaration-errors.m:[[@LINE-1]]:4: error: an equivalence type cannot have 'where' attributes
:- typeclass tc where [].
% CHECK: declaration-errors.m:[[@LINE-1]]:14: error: a type class must have at least one parameter
:- func f6(int).
% CHECK: declaration-errors.m:[[@LINE-1]]:9: error: expected a function's arguments and result, as in 'f(int) = int', but found 'f6(int)'
:- mode m7(in) : int.
% CHECK: declaration-errors.m:[[@LINE-1]]:18: error: expected a mode declaration without a type, but found 'int'
:- 42.
% CHECK: declaration-errors.m:[[@LINE-1]]:4: error: expected a declaration, but found '42'
:- type t8 ---> c8(int, 3).
% CHECK: declaration-errors.m:[[@LINE-1]]:25: error: expected a type, but found '3'
:- mode md10 == 42.
% CHECK: declaration-errors.m:[[@LINE-1]]:17: error: expected a mode, but found '42'
:- instance tc(int) where [ 42 ].
% CHECK: declaration-errors.m:[[@LINE-1]]:29: error: expected a clause head, but found '42'
