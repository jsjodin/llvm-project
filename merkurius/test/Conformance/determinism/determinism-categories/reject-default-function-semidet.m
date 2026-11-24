% MANUAL: Determinism categories
% A function without a mode declaration has the default mode, declared det,
% so a body that can fail is an error; a partial function must declare its
% mode and determinism.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_default_function_semidet.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- func half(int) = int.                            % ERROR
half(X) = X // 2 :-
    X mod 2 = 0.

main(!IO) :-
    io.write_line(half(4), !IO).
