% MANUAL: The module system
% A module whose entities the interface uses must be imported in the
% interface section.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_import_for_interface.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- pred first(list(int)::in, int::out) is semidet.  % ERROR
:- implementation.
:- import_module list.
first([X | _], X).
main(!IO) :-
    ( if first([1], X) then io.write_int(X, !IO) else true ).
