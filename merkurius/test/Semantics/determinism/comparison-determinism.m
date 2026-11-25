% A type's comparison predicate must be det (reference manual, "User-defined
% equality and comparison"): compare/3 at the type is det, so its procedure
% that gives the result and takes the values in may be det or erroneous, but
% must not fail or have more than one solution, as mmc's generated
% __Compare__ requires. Each error is at the type's definition. A predicate
% declared without a determinism is checked with the one inferred for it;
% a mode that does not give the result, as one that checks a given result,
% may have any.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --implicit-check-not=error: %s < %t.err

:- module comparison_determinism.
:- interface.
:- type unused ---> unused.
:- implementation.
:- import_module exception.

:- type c_multi ---> c_multi(int) where comparison is cmp_multi.
% CHECK: [[@LINE-1]]:9: error: the comparison predicate of type c_multi/0 must be det, but cmp_multi(uo, in, in) is multi
:- pred cmp_multi(comparison_result::uo, c_multi::in, c_multi::in) is multi.
cmp_multi(R, _, _) :- ( R = (=) ; R = (<) ).

:- type c_cc_multi ---> c_cc_multi(int) where comparison is cmp_cc_multi.
% CHECK: [[@LINE-1]]:9: error: the comparison predicate of type c_cc_multi/0 must be det, but cmp_cc_multi(uo, in, in) is cc_multi
:- pred cmp_cc_multi(comparison_result::uo, c_cc_multi::in, c_cc_multi::in) is cc_multi.
cmp_cc_multi(R, _, _) :- ( R = (=) ; R = (<) ).

:- type c_semidet ---> c_semidet(int) where comparison is cmp_semidet.
% CHECK: [[@LINE-1]]:9: error: the comparison predicate of type c_semidet/0 must be det, but cmp_semidet(uo, in, in) is semidet
:- pred cmp_semidet(comparison_result::uo, c_semidet::in, c_semidet::in) is semidet.
cmp_semidet(R, _, _) :- R = (=), semidet_true.

:- type c_failure ---> c_failure(int) where comparison is cmp_failure.
% CHECK: [[@LINE-1]]:9: error: the comparison predicate of type c_failure/0 must be det, but cmp_failure(uo, in, in) is failure
:- pred cmp_failure(comparison_result::uo, c_failure::in, c_failure::in) is failure.
cmp_failure(_, _, _) :- fail.

:- type c_nondet ---> c_nondet(int) where comparison is cmp_nondet.
% CHECK: [[@LINE-1]]:9: error: the comparison predicate of type c_nondet/0 must be det, but cmp_nondet(uo, in, in) is nondet
:- pred cmp_nondet(comparison_result::uo, c_nondet::in, c_nondet::in) is nondet.
cmp_nondet(R, _, _) :- ( R = (=) ; R = (<) ), semidet_true.

:- type c_cc_nondet ---> c_cc_nondet(int) where comparison is cmp_cc_nondet.
% CHECK: [[@LINE-1]]:9: error: the comparison predicate of type c_cc_nondet/0 must be det, but cmp_cc_nondet(uo, in, in) is cc_nondet
:- pred cmp_cc_nondet(comparison_result::uo, c_cc_nondet::in, c_cc_nondet::in) is cc_nondet.
cmp_cc_nondet(R, _, _) :- ( R = (=) ; R = (<) ), semidet_true.

:- type c_erroneous ---> c_erroneous(int) where comparison is cmp_erroneous.
:- pred cmp_erroneous(comparison_result::uo, c_erroneous::in, c_erroneous::in) is erroneous.
cmp_erroneous(_, _, _) :- throw(42).

:- type c_det ---> c_det(int) where comparison is cmp_det.
:- pred cmp_det(comparison_result::uo, c_det::in, c_det::in) is det.
cmp_det(R, _, _) :- R = (=).

:- type c_inferred_multi ---> c_inferred_multi(int)
    where comparison is cmp_inferred_multi.
% CHECK: [[@LINE-2]]:9: error: the comparison predicate of type c_inferred_multi/0 must be det, but cmp_inferred_multi(uo, in, in) is inferred multi
:- pred cmp_inferred_multi(comparison_result, c_inferred_multi,
    c_inferred_multi).
:- mode cmp_inferred_multi(uo, in, in).
cmp_inferred_multi(R, _, _) :- ( R = (=) ; R = (<) ).

:- type c_inferred_det ---> c_inferred_det(int)
    where comparison is cmp_inferred_det.
:- pred cmp_inferred_det(comparison_result, c_inferred_det, c_inferred_det).
:- mode cmp_inferred_det(uo, in, in).
cmp_inferred_det(R, _, _) :- R = (=).

:- type c_checking ---> c_checking(int) where comparison is cmp_checking.
:- pred cmp_checking(comparison_result, c_checking, c_checking).
:- mode cmp_checking(uo, in, in) is det.
:- mode cmp_checking(in, in, in) is semidet.
cmp_checking((=), _, _).
