% MANUAL: Default insts for functions
% A function term can be passed where no higher-order inst is expected
% only if it could be passed where the default func(in) = out is det is.
% A semidet one cannot.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:23:{{[0-9]+}}: error:

:- module reject_semidet_function_in_list.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- func halve(int) = int.
:- mode halve(in) = out is semidet.
halve(X) = X // 2 :-
    X mod 2 = 0.

main(!IO) :-
    Fs = [halve],
    list.length(Fs, N),                             % ERROR
    io.write_line(N, !IO).
