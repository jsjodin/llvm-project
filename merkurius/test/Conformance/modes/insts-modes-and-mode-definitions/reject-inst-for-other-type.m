% MANUAL: Insts modes and mode definitions
% An inst declared for values of one type constructor may not be applied
% to values of another.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:18:{{[0-9]+}}: error:

:- module reject_inst_for_other_type.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, maybe.

:- inst yes_only for maybe/1
    --->    yes(ground).

:- pred head(list(int)::in(yes_only), int::out) is det.  % ERROR
head(_, 0).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
