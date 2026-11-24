% MANUAL: Semantics of type quantifiers
% A caller may not use an existentially typed value as a value of the type
% the callee chose.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:22:{{[0-9]+}}: error:

:- module reject_use_as_known_type.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- some [T] pred e_foo(T).
:- mode e_foo(out) is det.
e_foo(42).

:- pred use(int::out) is det.
use(Y) :-
    e_foo(X),
    Y = X + 1.                                      % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
