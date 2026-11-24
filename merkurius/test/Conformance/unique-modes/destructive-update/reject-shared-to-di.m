% MANUAL: Destructive update
% A value with more than one reference is not unique, so it may not be
% passed to a di argument.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:18:{{[0-9]+}}: error:

:- module reject_shared_to_di.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- import_module array.

:- pred reset(array(int)::in, array(int)::out) is det.
reset(A0, A) :-
    array.set(0, 0, A0, A).                         % ERROR

main(!IO) :-
    array.init(2, 1, A0),
    reset(A0, A),
    array.to_list(A, L),
    io.write_line(L, !IO).
