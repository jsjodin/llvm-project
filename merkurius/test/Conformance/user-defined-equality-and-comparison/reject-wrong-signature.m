% MANUAL: User-defined equality and comparison
% The equality predicate must take two values of the type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_wrong_signature.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type box                                         % ERROR
    --->    box(int)
            where equality is int_equals.

:- pred int_equals(int::in, int::in) is semidet.
int_equals(X, X).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
