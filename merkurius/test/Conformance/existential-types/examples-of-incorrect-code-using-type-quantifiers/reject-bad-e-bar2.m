% MANUAL: Examples of incorrect code using type quantifiers
% Clauses may not bind an existential type to different types.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_bad_e_bar2.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- some [T] pred bad_e_bar2(T).
:- mode bad_e_bar2(out) is multi.
bad_e_bar2(42).
bad_e_bar2("blah").                                 % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
