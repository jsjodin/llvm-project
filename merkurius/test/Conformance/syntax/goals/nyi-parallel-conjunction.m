% MANUAL: Goals
% A parallel conjunction of det goals.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for procedures with parallel conjunctions, which mode analysis does not check yet

:- module nyi_parallel_conjunction.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

main(!IO) :-
    ( X = 1 & Y = 2 ),
    io.write_int(X + Y, !IO), io.nl(!IO).
