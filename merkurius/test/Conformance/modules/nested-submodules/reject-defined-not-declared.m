% MANUAL: Nested submodules
% A submodule may not be defined without being declared: an item sequence
% with only an implementation section defines it.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_defined_not_declared.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
    :- module child.                                % ERROR
    :- implementation.
    :- func one = int.
    one = 1.
    :- end_module child.

main(!IO) :-
    io.write_string("hello\n", !IO).
