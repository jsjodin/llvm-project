% MANUAL: Destructive update
% A value passed to a di argument is dead, so it may not be used
% afterwards: the first call may not destroy a value the second needs.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:18:{{[0-9]+}}: error:

:- module reject_use_after_di.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- import_module array.

main(!IO) :-
    array.init(2, 0, A0),
    array.set(0, 1, A0, A1),                        % ERROR
    array.set(1, 2, A0, A2),
    array.to_list(A1, L1),
    array.to_list(A2, L2),
    io.write_line({L1, L2}, !IO).
