% MANUAL: User-defined equality and comparison
% The equality predicate must be pure.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_not_pure.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type box                                         % ERROR
    --->    box(int)
            where equality is box_equals.

:- impure pred box_equals(box::in, box::in) is semidet.
box_equals(_, _) :-
    impure touch.

:- impure pred touch is semidet.
:- pragma foreign_proc("C", touch, [will_not_call_mercury],
    "SUCCESS_INDICATOR = MR_TRUE;").

main(!IO) :-
    io.write_string("unreachable\n", !IO).
