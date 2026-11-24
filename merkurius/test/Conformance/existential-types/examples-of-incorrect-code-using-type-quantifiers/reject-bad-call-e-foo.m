% MANUAL: Examples of incorrect code using type quantifiers
% A caller may not choose the type of an existentially quantified
% argument.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:20:{{[0-9]+}}: error:

:- module reject_bad_call_e_foo.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- some [T] pred e_foo(T).
:- mode e_foo(out) is det.
e_foo(42).
    % ok

:- pred bad_call_e_foo.
:- mode bad_call_e_foo is semidet.
bad_call_e_foo :- e_foo(42).                        % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
