% MANUAL: Implementing solver types
% A small solver: a variable with a lower bound, kept in its
% representation, constrained by impure foreign code, with a ground value
% converted to an int.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: the representation functions of solver types (representation of any bounded/0/1)

:- module nyi_bounded.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- solver type bounded where representation is int, equality is eq.
:- pred eq(bounded::ia, bounded::ia) is semidet.
:- pragma promise_pure(eq/2).
eq(A, B) :-
    impure RA = 'representation of any bounded/0'(A),
    impure RB = 'representation of any bounded/0'(B),
    RA = RB.
:- pred at_least(int::in, bounded::oa) is det.
:- pragma promise_pure(at_least/2).
at_least(N, B) :- impure B = 'representation to any bounded/0'(N).
:- impure pred bound(bounded::ia, int::out) is det.
bound(B, N) :- impure N = 'representation of any bounded/0'(B).

main(!IO) :-
    at_least(3, B),
    promise_pure ( impure bound(B, N) ),
    io.write_int(N, !IO), io.nl(!IO).
