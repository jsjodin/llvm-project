% A call whose output argument is already bound is in an implied mode: it
% outputs to a fresh value, which is tested against the bound one, so the
% call can fail. Calls of a function, of a predicate and of a built-in
% function, of int, float and term types, and with the bound argument an
% input too. The output is mmc's.
%
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace

:- module implied_modes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int.

:- type t
    --->    a
    ;       b(int).

:- func f(int) = int.
f(X) = X * 2.

:- pred g(int::in, int::out) is det.
g(X, X * 3).

:- func h(int) = t.
h(X) = ( if X > 0 then b(X) else a ).

:- pred t1(int::in) is semidet.
t1(A) :- f(A) = 6.

:- pred t2(int::in, int::in) is semidet.
t2(A, B) :- g(A, B).

:- pred t3(int::in, int::in) is semidet.
t3(A, B) :- max(A, 3) = B.

% The annotation makes B the call's own output variable.
:- pred t4(int::in, int::in) is semidet.
t4(A, B) :- B = (max(A, 3) : int).

:- pred t5(float::in, float::in) is semidet.
t5(A, B) :- B = (float.max(A, 1.5) : float).

:- pred t7(int::in) is semidet.
t7(A) :- A = A mod 3.

:- pred t8(int::in) is semidet.
t8(A) :- A = f(A).

:- pred t6(int::in, t::in) is semidet.
t6(A, T) :- T = (h(A) : t).

main(!IO) :-
    ( if t1(3) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t1(4) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t2(2, 6) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t2(2, 7) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t3(5, 5) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t3(5, 4) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t4(1, 3) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t4(1, 1) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t5(2.5, 2.5) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t5(0.5, 0.5) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t6(4, b(4)) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t6(4, a) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t6(0, a) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t7(2) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t7(4) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t8(0) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    ( if t8(1) then io.write_char('y', !IO) else io.write_char('n', !IO) ),
    io.nl(!IO).
% CHECK:ynynynynynynyynyn
