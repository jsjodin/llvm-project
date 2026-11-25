% MANUAL: Visibility rules
% A child's declarations are not visible in its parent unless the parent
% imports it.
% KNOWN-BUG: nested submodules are not yet implemented (docs/INPROGRESS.md, "Nested submodules").
% The rule for separate submodules: test/Semantics/modules/submodule-not-imported.m.
%
% XFAIL: *
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:25:{{[0-9]+}}: error:

:- module reject_child_not_imported.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.

    :- module child.
    :- interface.
    :- func peek = int.
    :- implementation.
    peek = 1.
    :- end_module child.

main(!IO) :-
    io.write_int(peek, !IO).                        % ERROR
