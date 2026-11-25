% MANUAL: The module system
% A module defines its own entities: the qualifier of a name that a
% declaration or definition introduces names the module. Each kind of
% defined name is given the qualifier Q; those qualified by the module are
% accepted.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:24:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:25:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:26:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:27:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:28:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:29:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:30:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:31:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:33:{{[0-9]+}}: error:

:- module reject_definition_qualifier.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
main(!IO) :- io.write_string("ok\n", !IO).
:- type 'Q'.t1 ---> a1. % ERROR
:- type t2 ---> 'Q'.c2. % ERROR
:- type t3 ---> c3('Q'.f :: int). % ERROR
:- inst 'Q'.i4 ---> a1. % ERROR
:- mode 'Q'.m5 == in. % ERROR
:- typeclass 'Q'.tc6(T) where [ pred meth6(T::in) is det ]. % ERROR
:- typeclass tc7(T) where [ pred 'Q'.meth7(T::in) is det ]. % ERROR
:- pred 'Q'.p8(int::in) is det. % ERROR
:- pred p9(int::in) is det.
:- mode 'Q'.p9(out) is det. % ERROR
p9(_).
:- type reject_definition_qualifier.t10 ---> a10.
:- pred reject_definition_qualifier.p11(int::in) is det.
reject_definition_qualifier.p11(_).
