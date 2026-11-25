% MANUAL: Feature sets
% parallel_conj and trailing may not both be required.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_parallel_conj_and_trailing.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pragma require_feature_set([parallel_conj, trailing]). % ERROR
main(!IO) :- io.write_string("ok\n", !IO).
