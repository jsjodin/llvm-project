% MANUAL: Nested submodules
% A submodule may not be declared twice.
% KNOWN-BUG: nested submodules are not yet implemented (docs/INPROGRESS.md, "Nested submodules").
%
% XFAIL: *
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:18:{{[0-9]+}}: error:

:- module reject_declared_twice.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

    :- module child.
    :- interface.
    :- type t ---> t.
    :- end_module child.

    :- module child.                                % ERROR
    :- interface.
    :- type u ---> u.
    :- end_module child.

:- implementation.
main(!IO) :-
    io.write_string("hello\n", !IO).
