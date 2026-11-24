% MANUAL: Creating higher-order terms
% A builtin such as \\= cannot be used as a higher-order term or
% curried.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_builtin_curried.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

main(!IO) :-
    list.filter(\=(2), [1, 2, 3], List),           % ERROR
    io.write_line(List, !IO).
