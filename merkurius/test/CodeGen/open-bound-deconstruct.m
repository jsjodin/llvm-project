% Deconstructions of open terms that are already bound (design document,
% "Open terms as built"): an open(T) argument of mode in is ground but still
% open words, read through a dereference, and a closed(T) hole bound earlier
% in the clause gives its value by unboxing it, with no walk of the term.
%
% RUN: merkc -fopen-types -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fopen-types -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module open_bound_deconstruct.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open.

:- type big ---> big(int, int).
:- type pair2 ---> p(closed(big), int).

:- type box ---> box(closed(big)).
:- pred hole(open(box)::oa) is det.
:- pragma no_inline(hole/1).
hole(box(_)).

:- pred hp(open(pair2)::oa) is det.
:- pragma no_inline(hp/1).
hp(p(_, 3)).

:- pred rd(open(pair2)::in, int::out) is semidet.
:- pragma no_inline(rd/2).
rd(p(closed(big(A, _)), N), A + N).

main(!IO) :-
    ( if rd(p(closed(big(5, 6)), 7), R) then
        io.write_int(R, !IO), io.nl(!IO)
    else
        io.write_string("no\n", !IO)
    ),
    promise_pure (
        hole(box(H)),
        ( if H = closed(big(1, 2)), H = closed(B) then
            io.write(B, !IO), io.nl(!IO)
        else
            io.write_string("no\n", !IO)
        ),
        hp(P),
        ( if P = p(closed(big(3, 4)), _), P = p(C, N), C = closed(big(X, Y)) then
            io.write(X + Y + N, !IO), io.nl(!IO)
        else
            io.write_string("no\n", !IO)
        )
    ).
% OUTPUT:12
% OUTPUT-NEXT:big(1, 2)
% OUTPUT-NEXT:10
