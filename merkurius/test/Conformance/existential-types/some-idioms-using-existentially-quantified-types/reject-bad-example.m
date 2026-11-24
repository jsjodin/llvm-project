% MANUAL: Some idioms using existentially quantified types
% An existentially typed predicate's clauses may not give its argument
% different types.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_bad_example.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- some [T] pred bad_example(string, T).
:- mode bad_example(in, out) is semidet.

bad_example("foo", 42).
bad_example("bar", "blah").                         % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
