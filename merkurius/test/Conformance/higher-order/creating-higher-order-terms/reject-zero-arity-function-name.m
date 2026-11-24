% MANUAL: Creating higher-order terms
% The name of a zero-arity function denotes its result, not the function,
% so applying it is a type error.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:20:{{[0-9]+}}: error:

:- module reject_zero_arity_function_name.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- func foo = int.
foo = 42.

main(!IO) :-
    F = foo,
    io.write_line(apply(F), !IO).                   % ERROR
