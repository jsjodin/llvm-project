% MANUAL: Examples of incorrect code using type quantifiers
% A universally quantified argument may not be bound to an int.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_bad_foo.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pred bad_foo(T).
:- mode bad_foo(in) is semidet.
bad_foo(42).                                        % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
