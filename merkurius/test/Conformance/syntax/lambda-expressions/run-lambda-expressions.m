% MANUAL: Lambda expressions
% Each form of lambda expression: pred with modes, pred with DCG modes,
% func with modes, func with a result in parentheses, func without, and
% one without a body; its arguments are local to it.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{7, [8]}
% OUTPUT-NEXT:{100, 2, 2, 3, 4, 1}
% OUTPUT-NOT:{{.}}

:- module run_lambda_expressions.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

main(!IO) :-
    X = 100,
    P = (pred(X::in, Y::out) is det :- Y = X + 1),
    D = (pred(M::out, in, out) is semidet --> [M]),
    F1 = (func(A::in) = (B::out) is det :- B = A * 2),
    F2 = (func(A) = (B) is det :- B = A * 3),
    F3 = (func(A) = B :- B = A * 4),
    F4 = (func(A) = A),
    P(1, R1),
    ( if D(N, [7, 8], Rest) then io.write({N, Rest}, !IO) else true ),
    io.nl(!IO),
    io.write({X, R1, F1(1), F2(1), F3(1), F4(1)}, !IO), io.nl(!IO).
