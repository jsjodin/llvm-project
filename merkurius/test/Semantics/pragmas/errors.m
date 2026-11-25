% obsolete, obsolete_proc and type_spec pragmas, checked as mmc checks them
% (reference manual, "Obsolescence", "Syntax and semantics of type
% specialization pragmas"): each names a predicate or function of the
% module, and a mode it has if it names one; a type_spec's substitution
% gives only the declaration's variables a type, each one type, which does
% not mention the variables replaced. A new variable on the right, and a
% replacement that does not exist, are accepted, as in mmc. (These run after
% mode analysis, past where compare-types stops; the conformance tests
% pragmas/... compare them with mmc's.)
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s

:- module errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

main(!IO) :-
    io.write_string("ok\n", !IO).

:- pred e(T::in, T::out) is det.
e(X, X).
:- pred f(T::in, U::in) is det.
f(_, _).

:- pragma type_spec(e/2, T = int).
:- pragma type_spec(e(in, out), T = list(U)).
:- pragma type_spec(e/2, (T = int, T = string)).
% CHECK: [[@LINE-1]]:4: error: the `:- pragma type_spec' substitution gives the variable T more than one type
:- pragma type_spec(e/2, T = list(T)).
% CHECK: [[@LINE-1]]:4: error: the variable T occurs on both sides of the `:- pragma type_spec' substitution
:- pragma type_spec(f/2, (T = int, U = list(T))).
% CHECK: [[@LINE-1]]:4: error: the variable T occurs on both sides of the `:- pragma type_spec' substitution
:- pragma type_spec(e/2, V = int).
% CHECK: [[@LINE-1]]:4: error: the variable V of the `:- pragma type_spec' substitution does not occur in the declaration of pred errors.e/2
:- pragma type_spec(e(out, in), T = int).
% CHECK: [[@LINE-1]]:4: error: `:- pragma type_spec' names a mode that pred errors.e/2 does not have
:- pragma type_spec(nosuch/2, T = int).
% CHECK: [[@LINE-1]]:4: error: `:- pragma type_spec' names nosuch/2, which this module does not declare

:- pragma obsolete(e/2, [nosuch/3]).
:- pragma obsolete(nosuch/1).
% CHECK: [[@LINE-1]]:4: error: `:- pragma obsolete' names nosuch/1, which this module does not declare
:- pragma obsolete(e/3).
% CHECK: [[@LINE-1]]:4: error: `:- pragma obsolete' names e/3, which this module does not declare; its arity is 2
:- pragma obsolete_proc(f(out, in)).
% CHECK: [[@LINE-1]]:4: error: `:- pragma obsolete_proc' names a mode that pred errors.f/2 does not have
