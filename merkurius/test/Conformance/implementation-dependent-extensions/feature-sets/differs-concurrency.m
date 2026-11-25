% MANUAL: Feature sets
% The concurrency feature, which merkc's one build has.
% DIFFERS: implementation: merkc has no grades, and its one build runs concurrent processes; mmc's hlc.gc grade has no threads, so it rejects the module
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:ok
% OUTPUT-NOT:{{.}}
% MMC-STATUS: rejected
% MMC: 21: Error: this module must be compiled in a grade that supports concurrent execution.

:- module differs_concurrency.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pragma require_feature_set([concurrency]).
main(!IO) :- io.write_string("ok\n", !IO).
