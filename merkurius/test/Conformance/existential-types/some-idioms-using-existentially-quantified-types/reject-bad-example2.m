% MANUAL: Some idioms using existentially quantified types
% Nor may the disjuncts of one clause.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_bad_example2.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- some [T] pred bad_example2(string, T).
:- mode bad_example2(in, out) is semidet.

bad_example2(Name, Value) :-
    ( Name = "foo", Value = 42
    ; Name = "bar", Value = "blah"                  % ERROR
    ).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
