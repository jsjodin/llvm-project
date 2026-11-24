% Errors in the attributes of foreign_proc (reference manual, "Foreign code
% attributes"): two of a group of which at most one may be given, and
% may_call_mm_tabled, which calls Merkurius, with will_not_call_mercury. An
% attribute may be repeated, and will_not_call_mm_tabled may be given with
% will_not_call_mercury, as mmc accepts (design document, the register).
%
% RUN: not merkc -fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module foreign_errors.
:- implementation.

:- pragma foreign_proc("C", p(X::out),
    [may_call_mercury, will_not_call_mercury], "X = 1;").
% CHECK: foreign-errors.m:[[@LINE-1]]:5: error: the attributes 'may_call_merkurius' and 'will_not_call_merkurius' conflict

:- pragma foreign_proc("C", q(X::out),
    [promise_pure, will_not_call_mercury, may_call_mm_tabled], "X = 1;").
% CHECK: foreign-errors.m:[[@LINE-1]]:5: error: the attributes 'will_not_call_merkurius' and 'may_call_mm_tabled' conflict

:- pragma foreign_proc("C", r(X::out),
    [promise_pure, will_not_call_mercury, will_not_call_mm_tabled,
        promise_pure], "X = 1;").
