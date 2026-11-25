% What merkc does with the extensions that mmc's installed grade, hlc.gc,
% rejects, so that the conformance suite cannot compare them (reference
% manual, "Tabled evaluation", "Feature sets"): minimal model tabling is not
% yet implemented, nor are the features trailing and parallel_conj, and
% merkc has no conservative collector. A require_feature_set may not be in
% the interface section, as the manual says; mmc accepts an empty one there
% (design document, "Where mmc and the manual disagree").
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s

:- module not_in_mmc_grade.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- pragma require_feature_set([]).
% CHECK: [[@LINE-1]]:4: error: `:- pragma require_feature_set' may not be in the interface section
:- implementation.
:- import_module solutions.

:- pred edge(int::out, int::out) is multi.
edge(1, 2).
edge(2, 3).
edge(3, 1).
:- pred path(int::out, int::out) is multi.
:- pragma minimal_model(path/2).
% CHECK: [[@LINE-1]]:4: error: not yet implemented: tabled evaluation (`:- pragma minimal_model')
path(X, Y) :- edge(X, Y).
path(X, Y) :- path(X, Z), edge(Z, Y).

:- pragma require_feature_set([trailing]).
% CHECK: [[@LINE-1]]:4: error: not yet implemented: trailing (the feature 'trailing')
:- pragma require_feature_set([parallel_conj]).
% CHECK: [[@LINE-1]]:4: error: not yet implemented: parallel conjunctions (the feature 'parallel_conj')
:- pragma require_feature_set([conservative_gc]).
% CHECK: [[@LINE-1]]:4: error: this module requires conservative garbage collection, and merkc's collector is precise

main(!IO) :-
    solutions(pred({X, Y}::out) is nondet :- path(X, Y), L),
    io.write(L, !IO),
    io.nl(!IO).
