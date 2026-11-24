% MANUAL: Builtin higher-order insts and modes
% The builtin higher-order insts and modes: a predicate that returns a
% higher-order term through free >> (pred(out) is det) or out(...), modes
% named by the insts, (pred) is det and func(in) = out is semidet, and the
% manual's comparison of two higher-order terms by their solutions.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{6, 4}
% OUTPUT-NEXT:"big"
% OUTPUT-NEXT:{3, -1}
% OUTPUT-NEXT:equivalent
% OUTPUT-NOT:{{.}}

:- module run_insts_and_modes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred sum(list(int)::in, int::out) is det.
sum(List, Sum) :-
    list.foldl((pred(E::in, A0::in, A::out) is det :- A = A0 + E), List, 0,
        Sum).

:- pred foo(pred(int)).
:- mode foo(free >> (pred(out) is det)) is det.
foo(sum([1, 2, 3])).

:- pred bar(pred(int)::out(pred(out) is det)) is det.
bar(sum([4])).

:- pred run(pred, string, string).
:- mode run((pred) is semidet, in, out) is det.
run(P, Name, Out) :-
    ( if P then Out = Name else Out = "not " ++ Name ).

:- pred half(func(int) = int, int, int).
:- mode half(func(in) = out is semidet, in, out) is det.
half(F, X, Y) :-
    ( if Y0 = F(X) then Y = Y0 else Y = -1 ).

:- func halve(int) = int.
:- mode halve(in) = out is semidet.
halve(X) = X // 2 :-
    X mod 2 = 0.

main(!IO) :-
    foo(P1),
    P1(X1),
    bar(P2),
    P2(X2),
    io.write_line({X1, X2}, !IO),
    run((pred) is semidet :- X1 > 5, "big", S),
    io.write_line(S, !IO),
    half(halve, 6, H1),
    half(halve, 7, H2),
    io.write_line({H1, H2}, !IO),
    P = (pred(X::out) is det :- X = 6),
    foo(Q),
    ( if all [X] (call(P, X) <=> call(Q, X)) then
        io.write_string("equivalent\n", !IO)
    else
        io.write_string("different\n", !IO)
    ).
