% MANUAL: Existentially typed data types
% A construction with an existentially quantified functor must be written
% with 'new '; without it, the functor is a deconstruction.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_without_new.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type any
    --->    some [T] any(T).

:- func make = any.
make = any(1).                                      % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
