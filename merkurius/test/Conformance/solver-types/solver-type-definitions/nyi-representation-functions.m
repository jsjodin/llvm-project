% MANUAL: Solver type definitions
% The manual's four functions between a solver type and its representation,
% and an equality predicate written with them.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: the representation functions of solver types (representation of any w/0/1)

:- module nyi_representation_functions.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- solver type w where representation is int, equality is eq_w.
:- pred eq_w(w::ia, w::ia) is semidet.
:- pragma promise_pure(eq_w/2).
eq_w(A, B) :-
    impure RA = 'representation of any w/0'(A),
    impure RB = 'representation of any w/0'(B),
    RA = RB.
:- pred w(int::in, w::oa) is det.
:- pragma promise_pure(w/2).
w(N, W) :- impure W = 'representation to any w/0'(N).
:- pred ground_w(int::in, w::out) is det.
:- pragma promise_pure(ground_w/2).
ground_w(N, W) :- impure W = 'representation to ground w/0'(N).
:- func value(w) = int.
:- pragma promise_pure(value/1).
value(W) = N :- impure N = 'representation of ground w/0'(W).

main(!IO) :-
    w(1, A), w(2, B),
    promise_pure (
        if A = B then io.write_string("equal\n", !IO)
        else io.write_string("not equal\n", !IO)
    ),
    ground_w(7, G), io.write_int(value(G), !IO), io.nl(!IO).
