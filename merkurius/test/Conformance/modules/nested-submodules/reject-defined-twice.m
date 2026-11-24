% MANUAL: Nested submodules
% A submodule may not be defined twice.
% KNOWN-BUG: nested submodules are not yet implemented (docs/INPROGRESS.md, "Nested submodules").
%
% XFAIL: *
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:24:{{[0-9]+}}: error:

:- module reject_defined_twice.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

    :- module child.
    :- interface.
    :- func one = int.
    :- end_module child.

:- implementation.
    :- module child.
    :- implementation.
    one = 1.
    :- end_module child.

    :- module child.                                % ERROR
    :- implementation.
    :- end_module child.

main(!IO) :-
    io.write_string("hello\n", !IO).
