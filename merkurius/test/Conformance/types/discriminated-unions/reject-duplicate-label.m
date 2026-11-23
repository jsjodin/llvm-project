% MANUAL: Discriminated unions
% It is an error for two fields in the same type to have the same label,
% whether in one constructor or in two.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:20:{{[0-9]+}}: error:

:- module reject_duplicate_label.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- type one
    --->    one(a :: int, a :: int).                % ERROR

:- type two
    --->    first(b :: int)
    ;       second(b :: string).                    % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
