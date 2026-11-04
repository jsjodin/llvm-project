% A condition that cannot succeed, here a test of a variable whose value
% mode analysis knows, leaves the variables after the if-then-else to the
% else part: they once had no values, which code generation reported as
% calls not implemented or crashed on. The output is mmc's.
%
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace

:- module condition_cannot_succeed.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- pred u(int::in, int::out) is semidet.
u(X, R) :-
    X = 2,
    R = ( if X = 3 then 15 else X ).

:- pred v(int::in, int::out) is semidet.
v(X, R) :-
    X = 2,
    ( if X = 3 then Y = 15 else Y = X ),
    R = (Y + 1) rem 7.

main(!IO) :-
    ( if u(2, R1) then io.write_int(R1, !IO) else io.write_string("no", !IO) ),
    io.nl(!IO),
    ( if v(2, R2) then io.write_int(R2, !IO) else io.write_string("no", !IO) ),
    io.nl(!IO),
    ( if v(4, R3) then io.write_int(R3, !IO) else io.write_string("no", !IO) ),
    io.nl(!IO).
% CHECK:2
% CHECK-NEXT:3
% CHECK-NEXT:no
