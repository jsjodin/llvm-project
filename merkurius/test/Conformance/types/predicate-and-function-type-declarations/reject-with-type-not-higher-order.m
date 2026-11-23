% MANUAL: Predicate and function type declarations
% The type after with_type, or :, must be a higher-order type of the same
% kind as the declaration.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_with_type_not_higher_order.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pred p : int.                                    % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
