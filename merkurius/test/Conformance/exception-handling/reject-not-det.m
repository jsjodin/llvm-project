% MANUAL: Exception handling
% A try goal has determinism cc_multi, which a det predicate cannot have.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:11:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_not_det.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det. % ERROR
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    ( try [] X = 1 % ERROR
    then
        io.write_int(X, !IO)
    ).
