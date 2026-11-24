% Solver type definitions (reference manual, "Solver type definitions"),
% checked as mmc checks them, though solver types are not implemented: a
% definition in the interface, which may only declare one abstractly; an
% abstract declaration and a definition that disagree on whether the type
% is a solver type, a foreign type included; a second definition, solver
% or not; and an abstract solver type with no definition. A solver type
% declared abstractly and defined is accepted. (mmc checks equality
% predicates only once these pass: conformance test
% solver-types/solver-type-definitions/reject-undefined-equality.m.)
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module solver_types.
:- interface.

:- solver type ok.

:- solver type in_interface where representation is int.
% CHECK: [[@LINE-1]]:16: error: the solver type in_interface/0 may be defined only in the implementation section; the interface may declare it abstractly

:- solver type union.
% CHECK: [[@LINE-1]]:16: error: the declaration of type solver_types.union/0 is of a solver type, but its definition is of a discriminated union type
% CHECK: because: the definition is here

:- solver type foreign.
% CHECK: [[@LINE-1]]:16: error: the declaration of type solver_types.foreign/0 is of a solver type, but its definition is of a foreign type
% CHECK: because: the definition is here

:- type not_solver.
% CHECK: [[@LINE-1]]:9: error: the declaration of type solver_types.not_solver/0 is not of a solver type, but its definition is
% CHECK: because: the definition is here

:- solver type undefined.
% CHECK: [[@LINE-1]]:16: error: the abstract type undefined/0 has no definition

:- implementation.

:- solver type ok where representation is int.
:- type union ---> a ; b.
:- pragma foreign_type("C", foreign, "MR_Integer").
:- solver type not_solver where representation is int.

:- solver type twice where representation is int.
:- solver type twice where representation is int.
% CHECK: [[@LINE-1]]:16: error: type solver_types.twice/0 is defined more than once
% CHECK: because: it is first defined here

:- solver type also_union where representation is int.
:- type also_union ---> c.
% CHECK: [[@LINE-1]]:9: error: type solver_types.also_union/0 is defined both as a solver type and as another type
% CHECK: because: the solver type is defined here
