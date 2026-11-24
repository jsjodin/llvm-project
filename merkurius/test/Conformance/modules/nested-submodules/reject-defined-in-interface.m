% MANUAL: Nested submodules
% A submodule may not be defined in the interface section of its
% parent.
% KNOWN-BUG: nested submodules are not yet implemented (docs/INPROGRESS.md, "Nested submodules").
%
% XFAIL: *
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_defined_in_interface.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

    :- module child.
    :- interface.
    :- func one = int.
    :- implementation.                              % ERROR
    one = 1.
    :- end_module child.

:- implementation.
main(!IO) :-
    io.write_string("hello\n", !IO).
