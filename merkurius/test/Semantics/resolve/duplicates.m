% Declarations that repeat an earlier one of the module, as mmc reports
% them: a predicate or function declared twice, whatever its types or
% modes, with `with_type`, or as a method too; a type, inst or mode
% defined twice, or a type as both a discriminated union type and an
% equivalence type; and, as a warning, a type declared abstractly twice.
% An abstract declaration or a foreign type goes with the definition.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module duplicates.
:- interface.
:- pred main(int::out) is det.
:- type abstract.
:- type abstract.
% CHECK: [[@LINE-1]]:9: warning: type duplicates.abstract/0 is declared abstractly more than once
% CHECK: [[@LINE-3]]:9: because: it is first declared here
:- type defined.
:- pred exported(int::in) is det.
:- implementation.
:- import_module int.

main(0).
:- type abstract ---> abstract.
:- type defined ---> defined.
:- type defined ---> again.
% CHECK: [[@LINE-1]]:9: error: type duplicates.defined/0 is defined more than once
% CHECK: [[@LINE-3]]:9: because: it is first defined here
:- pred exported(int::in) is det.
% CHECK: [[@LINE-1]]:9: error: pred duplicates.exported/1 is declared more than once
% CHECK: [[@LINE-12]]:9: because: it is first declared here
exported(_).

:- pred p(int::in) is det.
:- pred p(int::in) is det.
% CHECK: [[@LINE-1]]:9: error: pred duplicates.p/1 is declared more than once
% CHECK: [[@LINE-3]]:9: because: it is first declared here
:- pred p(string::in) is semidet.
% CHECK: [[@LINE-1]]:9: error: pred duplicates.p/1 is declared more than once
% CHECK: [[@LINE-6]]:9: because: it is first declared here
p(_).
:- func f(int) = int.
:- func f(int) = int.
% CHECK: [[@LINE-1]]:9: error: func duplicates.f/1 is declared more than once
% CHECK: [[@LINE-3]]:9: because: it is first declared here
f(X) = X.
% A predicate and a function of the same name and arity are different.
:- pred pf(int::in, int::out) is det.
:- func pf(int) = int.
pf(X, X).
pf(X) = X.
:- type pt == (pred(int)).
:- pred with_type `with_type` pt `with_inst` (pred(in) is det).
:- pred with_type(int::in) is det.
% CHECK: [[@LINE-1]]:9: error: pred duplicates.with_type/1 is declared more than once
% CHECK: [[@LINE-3]]:9: because: it is first declared here
with_type(_).

% A predicate first, then a method of that name and arity; the type class
% is then without it, as in mmc.
:- pred m(int::in) is det.
:- typeclass c(T) where [pred m(T::in) is det].
% CHECK: [[@LINE-1]]:31: error: pred duplicates.m/1 is declared more than once
% CHECK: [[@LINE-3]]:9: because: it is first declared here
m(_).

:- type t ---> a.
:- type t ---> b.
% CHECK: [[@LINE-1]]:9: error: type duplicates.t/0 is defined more than once
% CHECK: [[@LINE-3]]:9: because: it is first defined here
:- type e == int.
:- type e == int.
% CHECK: [[@LINE-1]]:9: error: type duplicates.e/0 is defined more than once
% CHECK: [[@LINE-3]]:9: because: it is first defined here
:- type de ---> de.
% CHECK: [[@LINE-1]]:9: error: type duplicates.de/0 is defined both as a discriminated union type and as an equivalence type
% CHECK: [[@LINE+1]]:9: because: the equivalence is defined here
:- type de == int.
:- type ed == int.
:- type ed ---> ed.
% CHECK: [[@LINE-1]]:9: error: type duplicates.ed/0 is defined both as a discriminated union type and as an equivalence type
% CHECK: [[@LINE-3]]:9: because: the equivalence is defined here
:- type foreign ---> foreign.
:- pragma foreign_type("C", foreign, "int").
:- inst i ---> a.
:- inst i ---> a.
% CHECK: [[@LINE-1]]:9: error: inst duplicates.i/0 is defined more than once
% CHECK: [[@LINE-3]]:9: because: it is first defined here
:- mode md == in.
:- mode md == in.
% CHECK: [[@LINE-1]]:9: error: mode duplicates.md/0 is defined more than once
% CHECK: [[@LINE-3]]:9: because: it is first defined here
