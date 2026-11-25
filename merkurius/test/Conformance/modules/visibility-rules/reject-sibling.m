% MANUAL: Visibility rules
% A child's declarations are not visible in its siblings unless they
% import it.
% KNOWN-BUG: nested submodules are not yet implemented (docs/INPROGRESS.md, "Nested submodules").
% The rule for separate submodules: test/Semantics/modules/submodule-sibling.m.
%
% XFAIL: *
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:30:{{[0-9]+}}: error:

:- module reject_sibling.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module reject_sibling.second.

    :- module first.
    :- interface.
    :- func one = int.
    :- implementation.
    one = 1.
    :- end_module first.

    :- module second.
    :- interface.
    :- func two = int.
    :- implementation.
    :- import_module int.
    two = one + 1.                                  % ERROR
    :- end_module second.

main(!IO) :-
    io.write_int(two, !IO).
