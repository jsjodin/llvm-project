% MANUAL: Module-local mutable variables
% A mutable's initial value is set as though by an initialise directive in
% its place: after the initialisers before it and before those after
% it.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: desugaring mutable declarations

:- module nyi_initial_order.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- initialise foo/2.
:- mutable(bar, int, 561, ground, [untrailed]).
:- initialise baz/2.

:- pred foo(io::di, io::uo) is det.
foo(!IO) :-
    io.write_string("foo\n", !IO).

:- pred baz(io::di, io::uo) is det.
baz(!IO) :-
    promise_pure (
        semipure get_bar(B)
    ),
    io.write_line(B, !IO).

main(!IO) :-
    io.write_string("main\n", !IO).
