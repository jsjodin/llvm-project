% MANUAL: Determinism categories
% A mode of a function whose arguments are all input can only be det,
% semidet, erroneous or failure.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_function_multi.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- func either(int) = int.
:- mode either(in) = out is multi.                  % ERROR
either(X) = Y :-
    ( Y = X ; Y = X + 1 ).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
