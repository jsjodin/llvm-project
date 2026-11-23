% MANUAL: Subtypes
% The chain of supertypes of a subtype may not be circular.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_circular_chain.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- type a =< b                                      % ERROR
    --->    x.
:- type b =< a                                      % ERROR
    --->    x.

main(!IO) :-
    io.write_string("unreachable\n", !IO).
