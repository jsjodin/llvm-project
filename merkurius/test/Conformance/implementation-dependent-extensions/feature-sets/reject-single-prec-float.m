% MANUAL: Feature sets
% The single_prec_float feature, which neither mmc's grade nor merkc has.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_single_prec_float.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pragma require_feature_set([single_prec_float]). % ERROR
main(!IO) :- io.write_string("ok\n", !IO).
