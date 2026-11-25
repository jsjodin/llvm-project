% MANUAL: Type conversion expressions
% coerce(Term) to a supertype.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for values of type nyi_coerce.pome

:- module nyi_coerce.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type fruit ---> apple ; pear ; plum.
:- type pome =< fruit ---> apple ; pear.

main(!IO) :-
    P = pear : pome,
    F = coerce(P) : fruit,
    io.write(F, !IO), io.nl(!IO).
