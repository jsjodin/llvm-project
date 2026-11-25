% MANUAL: Unique modes
% MANUAL: Predicate and function type declarations
% Interaction: an array updated in place in unique modes, in polymorphic
% procedures that carry the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 0 1 4 9 | 0 1 4 9 16 25
% OUTPUT-NEXT:2: 0 1 4 9 16 | 0 1 4 9 16 25 36
% OUTPUT-NEXT:3: 0 1 4 9 16 25 | 0 1 4 9 16 25 36 49
% OUTPUT-NEXT:4: 0 1 4 9 16 25 36 | 0 1 4 9 16 25 36 49 64
% OUTPUT-NOT:{{.}}
:- module polymorphic_unique_modes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, string.

:- type fdata == int.

:- func make(int) = fdata.
:- func use(fdata) = string.

make(N) = N + 3.

use(Size) = string.join_list(" ",
    list.map(string.int_to_string, array.to_list(squares(0, A0)))) :-
    array.init(Size, 0, A0).

    % Updates the array in place, as its modes allow.
:- func squares(int, array(int)) = array(int).
:- mode squares(in, array_di) = array_uo is det.

squares(I, A0) = A :-
    ( if I < array.size(A0) then
        array.set(I, I * I, A0, A1),
        A = squares(I + 1, A1)
    else
        A = A0
    ).

    % Polymorphic code that carries the values, in lists, pairs and maybes,
    % compiled once for every type.
:- pred keep(T::in, T::out) is det.
:- pragma no_inline(keep/2).

keep(X, X).

:- type holder(T)
    --->    holder(T, int)
    ;       nothing.

:- func second(list(T), T) = T.
:- pragma no_inline(second/2).

second(Xs, Default) =
    ( if Xs = [_, X | _] then X else Default ).

:- func run(int) = string.

run(N) = S :-
    keep([make(N), make(N + 1)], Vs),
    keep(holder(make(N + 2), N), H),
    ( if H = holder(V, _) then
        Held = use(V)
    else
        Held = "nothing"
    ),
    S = use(second(list.reverse(Vs), make(0))) ++ " | " ++ Held.

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
