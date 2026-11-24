% MANUAL: Examples of incorrect code using type quantifiers
% Two clauses that each take the type from another existentially typed
% call bind the type variable twice.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:20:{{[0-9]+}}: error:

:- module reject_bad_e_bar3.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- some [T] pred e_foo(T).
:- mode e_foo(out) is det.
e_foo(42).

:- some [T] pred bad_e_bar3(T).
:- mode bad_e_bar3(out) is multi.
bad_e_bar3(X) :- e_foo(X).
bad_e_bar3(X) :- e_foo(X).                          % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
