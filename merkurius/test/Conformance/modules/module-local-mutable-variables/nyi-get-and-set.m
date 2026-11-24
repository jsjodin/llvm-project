% MANUAL: Module-local mutable variables
% A mutable's get and set predicates, semipure and impure. Untrailed, as
% mmc's grades without trailing reject trailed mutables.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: desugaring mutable declarations

:- module nyi_get_and_set.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- mutable(counter, int, 0, ground, [untrailed]).
:- mutable(name, string, "start", ground, [untrailed]).

:- impure pred bump is det.
bump :-
    semipure get_counter(C),
    impure set_counter(C + 1).

main(!IO) :-
    promise_pure (
        impure bump,
        impure bump,
        semipure get_counter(C),
        impure set_name("end"),
        semipure get_name(N)
    ),
    io.write_line({C, N}, !IO).
