% MANUAL: Combined higher-order types and insts
% A higher-order type with modes and a determinism is allowed only as a
% direct argument of a constructor: not in a predicate's or function's
% declaration.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_in_declaration.
:- interface.
:- import_module io.

:- pred p((pred(io::di, io::uo) is det)::in, io::di, io::uo) is det. % ERROR
:- func f(func(int::in) = (int::out) is semidet, int) = int. % ERROR

:- implementation.

p(_, !IO).
f(_, X) = X.
