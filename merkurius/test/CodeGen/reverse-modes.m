% The reverse modes of int's and uint8's +, - and xor, X + Y = R with X or Y the
% output, which unifications of two applications can need: in
% A + B = C + D the second + is called with its result bound, before D,
% here a call, is computed, as in mmc. The output is mmc's.
%
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace

:- module reverse_modes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, uint8.

:- func sq(int) = int.
sq(X) = X * X.

:- pred plus(int::in, int::in) is semidet.
plus(A, B) :- A + B = sq(A) + sq(B).

:- pred minus(int::in, int::in) is semidet.
minus(A, B) :- A - B = sq(A) - sq(B).

:- pred minus_left(int::in, int::in) is semidet.
minus_left(A, B) :- A - B = sq(B) - A.

:- pred exclusive(int::in, int::in) is semidet.
exclusive(A, B) :- xor(A, B) = xor(sq(A), sq(B)).

:- func sq8(uint8) = uint8.
sq8(X) = X * X.

% With wrapping, 3 - 254 = 9 - 254 * 254, as 5 = 5 modulo 256.
:- pred small(uint8::in, uint8::in) is semidet.
small(A, B) :- A - B = sq8(A) - sq8(B).

:- pred yn(bool::in, io::di, io::uo) is det.
:- type bool ---> yes ; no.
yn(yes, !IO) :- io.write_char('y', !IO).
yn(no, !IO) :- io.write_char('n', !IO).

main(!IO) :-
    ( if plus(0, 1) then yn(yes, !IO) else yn(no, !IO) ),
    ( if plus(2, 3) then yn(yes, !IO) else yn(no, !IO) ),
    ( if minus(1, 0) then yn(yes, !IO) else yn(no, !IO) ),
    ( if minus(3, 2) then yn(yes, !IO) else yn(no, !IO) ),
    ( if minus(-3, 2) then yn(yes, !IO) else yn(no, !IO) ),
    ( if minus_left(2, 2) then yn(yes, !IO) else yn(no, !IO) ),
    ( if minus_left(3, 2) then yn(yes, !IO) else yn(no, !IO) ),
    ( if exclusive(1, 0) then yn(yes, !IO) else yn(no, !IO) ),
    ( if exclusive(2, 3) then yn(yes, !IO) else yn(no, !IO) ),
    ( if small(3u8, 254u8) then yn(yes, !IO) else yn(no, !IO) ),
    ( if small(3u8, 253u8) then yn(yes, !IO) else yn(no, !IO) ),
    io.nl(!IO).
% CHECK:ynynnnyynyn
