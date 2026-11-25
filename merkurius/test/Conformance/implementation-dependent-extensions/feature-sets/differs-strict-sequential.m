% MANUAL: Feature sets
% The strict_sequential feature, which merkc has (design document,
% "Operational semantics").
% DIFFERS: implementation: merkc gives only the strict sequential semantics; mmc's default is the strict commutative one, so it rejects the module unless asked for strict sequential
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:ok
% OUTPUT-NOT:{{.}}
% MMC-STATUS: rejected
% MMC: 22: Error: this module must be compiled using the strict sequential semantics.

:- module differs_strict_sequential.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pragma require_feature_set([strict_sequential]).
main(!IO) :- io.write_string("ok\n", !IO).
