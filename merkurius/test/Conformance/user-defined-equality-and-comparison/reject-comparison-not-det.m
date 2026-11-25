% MANUAL: User-defined equality and comparison
% The comparison predicate must be det: compare/3 at the type is det, so
% a comparison predicate that can fail or has more than one solution is an
% error, unlike an equality predicate, which may be nondet.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:
:- module reject_comparison_not_det.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t                                           % ERROR
    --->    t(int)
            where equality is t_equal, comparison is t_compare.
:- pred t_equal(t::in, t::in) is semidet.
t_equal(_, _) :- semidet_true.
:- pred t_compare(comparison_result::uo, t::in, t::in) is multi.
t_compare(R, _, _) :- ( R = (=) ; R = (<) ).
main(!IO) :-
    compare(R, t(2), t(1)),
    io.write(R, !IO),
    io.nl(!IO).
