% MANUAL: Module initialisation
% The predicate of an initialise directive must be declared, and as a pure
% pred(io::di, io::uo) or an impure predicate without arguments, det or
% cc_multi: not semidet, not of other types, not pure without arguments,
% not impure with I/O states, and not a function.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:20:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:21:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:22:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:23:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:24:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:25:{{[0-9]+}}: error:

:- module reject_signatures.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- initialise missing/2.                            % ERROR
:- initialise semi/0.                               % ERROR
:- initialise wrong_type/2.                         % ERROR
:- initialise not_impure/0.                         % ERROR
:- initialise impure_io/2.                          % ERROR
:- initialise a_function/0.                         % ERROR
:- initialise two_modes/2.

:- impure pred semi is semidet.
semi :-
    impure impure_true,
    semidet_true.

:- pred wrong_type(int::in, int::out) is det.
wrong_type(X, X).

:- pred not_impure is det.
not_impure.

:- impure pred impure_io(io::di, io::uo) is det.
impure_io(!IO) :-
    impure impure_true.

:- func a_function = int.
a_function = 1.

:- pred two_modes(io, io).
:- mode two_modes(di, uo) is det.
:- mode two_modes(mdi, muo) is det.
two_modes(!IO).

main(!IO) :-
    io.write_string("main\n", !IO).
