% MANUAL: Some idioms using existentially quantified types
% Nor may mode-specific clauses.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_bad_example3.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- some [T] pred bad_example3(string, T).
:-         mode bad_example3(in(bound("foo")), out) is det.
:-          mode bad_example3(in(bound("bar")), out) is det.
:- pragma promise_pure(bad_example3/2).
bad_example3("foo"::in(bound("foo")), 42::out).
bad_example3("bar"::in(bound("bar")), "blah"::out). % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
