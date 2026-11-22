% The collector copies along a cell's last field in a loop when that field
% has the cell's own type and its other fields hold no pointers, as a
% list(int)'s do, and pushes the fields as pending work otherwise. Each
% shape is live across collections here, at every allocation and at the
% heap's usual pace: lists of ints, floats, chars and an enum; a chain of
% two functors that alternate, and one whose type has nine functors, so
% that cells share a tag and have a word before their fields; and lists
% whose heads are pointers, which take the pending path. The output is
% mmc's.
%
% RUN: merkc -O2 %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: env MERK_GC_EVERY_ALLOC=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -O0 %s -o %t.O0 && env MERK_GC_EVERY_ALLOC=1 %t.O0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines

:- module gc_loops.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, float, int, list, string.

:- type colour ---> red ; green ; blue.

    % Two functors that alternate along the chain.
:- type chain ---> a(int, chain) ; b(char, chain) ; end.

    % Nine functors with arguments: the last ones share a tag.
:- type nine
    --->    n0(int, nine) ; n1(int, nine) ; n2(int, nine) ; n3(int, nine)
    ;       n4(int, nine) ; n5(int, nine) ; n6(int, nine) ; n7(int, nine)
    ;       n8(int, nine) ; stop.

:- func chain(int) = chain.
chain(N) = C :-
    ( if N = 0 then
        C = end
    else if N mod 2 = 0 then
        C = a(N, chain(N - 1))
    else
        C = b(char.det_from_int(0'a + N mod 26), chain(N - 1))
    ).

:- func chain_sum(chain, int) = int.
chain_sum(end, S) = S.
chain_sum(a(N, C), S) = chain_sum(C, S + N).
chain_sum(b(Ch, C), S) = chain_sum(C, S + char.to_int(Ch)).

:- func nine(int) = nine.
nine(N) = T :-
    ( if N = 0 then
        T = stop
    else
        R = nine(N - 1),
        K = N mod 9,
        ( if K = 0 then T = n0(N, R)
        else if K = 1 then T = n1(N, R)
        else if K = 2 then T = n2(N, R)
        else if K = 3 then T = n3(N, R)
        else if K = 4 then T = n4(N, R)
        else if K = 5 then T = n5(N, R)
        else if K = 6 then T = n6(N, R)
        else if K = 7 then T = n7(N, R)
        else T = n8(N, R)
        )
    ).

    % Each functor's number times its argument, so that a functor copied
    % as another changes the sum.
:- func nine_sum(nine, int) = int.
nine_sum(stop, S) = S.
nine_sum(n0(N, T), S) = nine_sum(T, S + 0 * N).
nine_sum(n1(N, T), S) = nine_sum(T, S + 1 * N).
nine_sum(n2(N, T), S) = nine_sum(T, S + 2 * N).
nine_sum(n3(N, T), S) = nine_sum(T, S + 3 * N).
nine_sum(n4(N, T), S) = nine_sum(T, S + 4 * N).
nine_sum(n5(N, T), S) = nine_sum(T, S + 5 * N).
nine_sum(n6(N, T), S) = nine_sum(T, S + 6 * N).
nine_sum(n7(N, T), S) = nine_sum(T, S + 7 * N).
nine_sum(n8(N, T), S) = nine_sum(T, S + 8 * N).

:- func colour(int) = colour.
colour(N) = ( if N mod 3 = 0 then red else if N mod 3 = 1 then green else blue ).

:- func count_blue(list(colour)) = int.
count_blue(Cs) = list.length(list.filter((pred(C::in) is semidet :- C = blue), Cs)).

    % Allocates N cells that die at once, which collects along the way.
:- func churn(int) = int.
:- pragma no_inline(churn/1).
churn(N) = list.length(list.duplicate(N, N)).

main(!IO) :-
    N = 20000,
    Ints = 1 .. N,
    Floats = list.map(float.float, Ints),
    Chars = list.map((func(I) = char.det_from_int(0'a + I mod 26)), Ints),
    Colours = list.map(colour, Ints),
    Chain = chain(N),
    Nine = nine(N),
    Lists = list.map((func(I) = [I, I + 1]), 1 .. 2000),
    Strings = list.map(string.int_to_string, 1 .. 2000),
    X = churn(100000),
    io.write_int(X, !IO), io.nl(!IO),
    io.write_int(list.foldl(plus, Ints, 0), !IO), io.nl(!IO),
    io.write_float(list.foldl((func(F, A) = F + A), Floats, 0.0), !IO), io.nl(!IO),
    io.write_string(string.from_char_list(list.take_upto(30, Chars)), !IO),
    io.nl(!IO),
    io.write_int(count_blue(Colours), !IO), io.nl(!IO),
    io.write_int(chain_sum(Chain, 0), !IO), io.nl(!IO),
    io.write_int(nine_sum(Nine, 0), !IO), io.nl(!IO),
    io.write_int(list.foldl(plus,
        list.map((func(L) = list.foldl(plus, L, 0)), Lists), 0), !IO),
    io.nl(!IO),
    io.write_string(string.join_list(",", list.take_upto(5, Strings)), !IO),
    io.write_string(" ", !IO),
    io.write_int(list.length(Strings), !IO), io.nl(!IO).

% OUTPUT: 100000
% OUTPUT-NEXT: 200010000
% OUTPUT-NEXT: 200010000.0
% OUTPUT-NEXT: bcdefghijklmnopqrstuvwxyzabcde
% OUTPUT-NEXT: 6667
% OUTPUT-NEXT: 101109970
% OUTPUT-NEXT: 799993331
% OUTPUT-NEXT: 4004000
% OUTPUT-NEXT: 1,2,3,4,5 2000
