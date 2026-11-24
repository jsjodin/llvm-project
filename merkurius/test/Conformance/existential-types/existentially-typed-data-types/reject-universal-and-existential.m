% MANUAL: Existentially typed data types
% No type variable may be quantified both universally, as a parameter of
% the type, and existentially, before a constructor.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_universal_and_existential.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t12(T)
    --->    f1(T)
    ;       some [T] f2(T).                         % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
