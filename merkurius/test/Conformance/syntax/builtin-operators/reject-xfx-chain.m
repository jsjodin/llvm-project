% MANUAL: Builtin operators
% An xfx operator's arguments have a strictly lower priority, so '='
% does not chain.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_xfx_chain.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
main(!IO) :- ( if 1 = 1 = 1 then true else true ), io.write_string("ok", !IO). % ERROR
