% MANUAL: Nested submodules
% A submodule declared but not defined has an empty implementation, which
% is an error when its interface declares a function.
% KNOWN-BUG: nested submodules are not yet implemented (docs/INPROGRESS.md, "Nested submodules").
%
% XFAIL: *
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_implicit_empty_implementation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

    :- module child.
    :- interface.
    :- func one = int.                              % ERROR
    :- end_module child.

:- implementation.
main(!IO) :-
    io.write_string("hello\n", !IO).
