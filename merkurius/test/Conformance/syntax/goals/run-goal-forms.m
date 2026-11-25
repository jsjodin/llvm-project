% MANUAL: Goals
% Each form of goal: quantifications, conjunction, disjunction, true,
% fail, both negations, the implications, both if-then-else syntaxes,
% inequality, call/N, a variable's call, purity and determinism casts,
% the determinism and switch checks, and disable_warnings in both forms.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{yes, yes, yes, yes, yes, no, yes, yes, 2, 5, 1, 2}
% OUTPUT-NOT:{{.}}

:- module run_goal_forms.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type c ---> r ; g ; b.
:- pred small(int::in) is semidet.
small(X) :- X < 3.
:- pred colour(c::in, int::out) is det.
colour(C, N) :-
    require_complete_switch [C]
    ( C = r, N = 1
    ; C = g, N = 2
    ; C = b, N = 3
    ).
:- pred colour2(c::in, int::out) is det.
colour2(C, N) :-
    require_switch_arms_det [C]
    ( C = r, N = 1
    ; C = g, N = 2
    ; C = b, N = 3
    ).
:- pred any_of(int::out) is multi.
any_of(1).
any_of(1).

main(!IO) :-
    L = [1, 2, 5],
    B1 = ( if some [X] (list.member(X, L), X > 4) then yes else no ),
    B2 = ( if all [X] (list.member(X, L) => X > 0) then yes else no ),
    B3 = ( if (true, (fail ; true)) then yes else no ),
    B4 = ( if not small(5), \+ small(4) then yes else no ),
    B5 = ( if small(1) <= small(0), (small(1) <=> small(2)) then yes else no ),
    B6 = ( small(9) -> yes ; no ),
    B7 = ( if 1 \= 2 then yes else no ),
    P = small,
    B8 = ( if call(P, 1), P(2), call(small, 0) then yes else no ),
    promise_equivalent_solutions [Y] any_of(Y),
    promise_pure ( require_det ( Z = Y + 1 ) ),
    colour(g, N0),
    colour2(b, N1),
    N = N0 + N1,
    disable_warnings [singleton_vars] ( W = 1 ),
    disable_warning [singleton_vars] ( V = 2 ),
    io.write({B1, B2, B3, B4, B5, B6, B7, B8, Z, N, W, V}, !IO), io.nl(!IO).
:- import_module bool.
