% MANUAL: Module-local mutable variables
% attach_to_io_state gives get and set predicates that thread the I/O
% state.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: desugaring mutable declarations

:- module nyi_attach_to_io_state.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- mutable(total, int, 10, ground, [untrailed, attach_to_io_state]).

main(!IO) :-
    get_total(T0, !IO),
    set_total(T0 * 2, !IO),
    get_total(T, !IO),
    io.write_line(T, !IO).
