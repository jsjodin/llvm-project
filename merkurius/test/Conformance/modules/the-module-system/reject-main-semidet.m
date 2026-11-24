% MANUAL: The module system
% main/2 must be declared det or cc_multi, with io.state arguments in
% modes di and uo.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:11:{{[0-9]+}}: error:

:- module reject_main_semidet.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is semidet.            % ERROR
:- implementation.
main(!IO) :-
    io.write_string("hello\n", !IO),
    semidet_true.
