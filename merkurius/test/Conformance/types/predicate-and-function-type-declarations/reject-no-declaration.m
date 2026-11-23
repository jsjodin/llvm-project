% MANUAL: Predicate and function type declarations
% Each predicate and function must have a type declaration. merkc infers
% the types of local ones by default, as mmc does with --infer-types, and
% reports them with -fno-infer-types (design document, "Type inference").
% MERKC-FLAGS: -fno-infer-types
%
% RUN: not merkc -fno-infer-types %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_no_declaration.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

double(X) = X * 2.                                  % ERROR

main(!IO) :-
    io.write_int(double(2), !IO),
    io.nl(!IO).
