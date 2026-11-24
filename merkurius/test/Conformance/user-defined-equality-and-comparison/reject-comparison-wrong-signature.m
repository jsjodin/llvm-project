% MANUAL: User-defined equality and comparison
% The comparison predicate must have the signature of compare/3 at the
% type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_comparison_wrong_signature.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type box                                         % ERROR
    --->    box(int)
            where comparison is box_compare.

:- pred box_compare(box::in, box::in, comparison_result::uo) is det.
box_compare(_, _, (=)).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
