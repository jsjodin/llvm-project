% MANUAL: Module finalisation
% The predicate of a finalise directive has the signatures an initialise
% directive's may have.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_signature.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- finalise semi/2.                                 % ERROR

:- pred semi(int::in, int::out) is semidet.
semi(X, X) :-
    semidet_true.

main(!IO) :-
    io.write_string("main\n", !IO).
