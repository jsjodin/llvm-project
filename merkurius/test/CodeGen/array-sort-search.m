% array.sort, the foldr family, binary_search and approx_binary_search with
% the standard order and a comparison function, and append, which the
% library compiles from array.m: sorts of up to a thousand elements, already
% sorted, reversed and with duplicates, of ints, narrow ints, floats,
% strings, characters and terms; folds in det and semidet modes and with io
% as the accumulator, which shows the order of the calls; searches for every
% value around arrays with duplicates, where binary_search finds the copy
% that mmc's does; and appends with either side empty. The output is mmc's.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT:sort 0: 0 sorted
% OUTPUT-NEXT:sort 1: 8 sorted
% OUTPUT-NEXT:sort 2: 257 sorted
% OUTPUT-NEXT:sort 3: 5151 sorted
% OUTPUT-NEXT:sort 7: 125220114 sorted
% OUTPUT-NEXT:sort 8: 359350338 sorted
% OUTPUT-NEXT:sort 9: 107746466 sorted
% OUTPUT-NEXT:sort 64: 72008277 sorted
% OUTPUT-NEXT:sort 100: 640894546 sorted
% OUTPUT-NEXT:sort 1000: 216861152 sorted
% OUTPUT-NEXT:array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
% OUTPUT-NEXT:array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
% OUTPUT-NEXT:array([1, 1, 2, 3, 3, 3])
% OUTPUT-NEXT:array(["", "apple", "apple", "fig", "pear"])
% OUTPUT-NEXT:array([-7.25, -1.0, 0.0, 2.5, 2.5])
% OUTPUT-NEXT:array([-128i8, -1i8, 0i8, 3i8, 127i8])
% OUTPUT-NEXT:array(['A', 'a', 'z', 'é'])
% OUTPUT-NEXT:array([circle(-1.0), circle(1.5), rect(1, 9), rect(2, 1)])
% OUTPUT-NEXT:4321
% OUTPUT-NEXT:[1, 2, 3, 4]
% OUTPUT-NEXT:[4, 3, 2, 1]
% OUTPUT-NEXT:4321
% OUTPUT-NEXT:10
% OUTPUT-NEXT:foldr failed at the first
% OUTPUT-NEXT:{4321, 10}
% OUTPUT-NEXT:4321{4321, 10}
% OUTPUT-NEXT:{10, 24, [1, 2, 3, 4], "4321", 4}
% OUTPUT-NEXT:7
% OUTPUT-NEXT:"cba"
% OUTPUT-NEXT:-1:no/no 0:no/no 1:0/0 2:no/0 3:2/2 4:no/3 5:4/4 6:no/4 7:5/5 8:no/6 9:8/8 10:no/10 11:11/11 12:no/11
% OUTPUT-NEXT:0:no/no 1:no/no
% OUTPUT-NEXT:3:no/no 4:0/0 5:no/0
% OUTPUT-NEXT:1:no/no 2:3/3 3:no/6
% OUTPUT-NEXT:-1:no/6 0:no/6 1:6/6 2:no/5 3:5/5 4:no/3 5:3/3 6:no/2 7:1/1 8:no/0 9:0/0 10:no/no
% OUTPUT-NEXT:0:0/0 1:5/5 2:11/11 3:17/17 4:20/20 5:24/24 6:30/30 7:35/35 8:36/36 9:42/42 10:45/45 11:47/47 12:49/49 13:50/50 14:52/52 15:55/55 16:58/58 17:61/61 18:67/67 19:74/74 20:77/77 21:80/80 22:85/85 23:86/86 24:90/90 25:92/92 26:95/95 27:99/99 28:102/102 29:103/103 30:105/105 31:111/111 32:117/117 33:119/119 34:120/120 35:124/124 36:130/130 37:133/133 38:136/136 39:142/142 40:149/149 41:152/152 42:153/153 43:155/155 44:161/161 45:167/167 46:168/168 47:170/170 48:174/174 49:177/177 50:180/180 51:183/183 52:185/185 53:187/187 54:188/188 55:193/193 56:194/194 57:196/196 58:198/198 59:199/199 60:no/199
% OUTPUT-NEXT::no ~no
% OUTPUT-NEXT:a:0 ~0
% OUTPUT-NEXT:b:no ~0
% OUTPUT-NEXT:c:1 ~1
% OUTPUT-NEXT:d:no ~2
% OUTPUT-NEXT:e:3 ~3
% OUTPUT-NEXT:f:no ~3
% OUTPUT-NEXT:array([1, 2, 3, 4, 5])
% OUTPUT-NEXT:array([3])
% OUTPUT-NEXT:array([1])
% OUTPUT-NEXT:array([])
% OUTPUT-NEXT:array([circle(1.0), rect(1, 2)])
% OUTPUT-NEXT:array([1.5, 2.5, 3.5])
% OUTPUT-NEXT:array([1i8, -2i8])
% OUTPUT-NEXT:{1200, 563689036, 848927994}
% OUTPUT-NOT:{{.}}

:- module array_sort_search.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, char, float, int, list, string.
:- use_module int8.

:- type shape
    --->    circle(float)
    ;       rect(int, int).

    % Pseudo-random numbers below M, from a linear congruence.
:- func numbers(int, int, int) = list(int).
numbers(N, M, Seed) =
    ( if N = 0 then []
    else [Seed mod M |
        numbers(N - 1, M, (Seed * 1103515245 + 12345) mod 2147483648)] ).

:- func checksum(array(int)) = int.
checksum(A) = array.foldl((func(X, S) = (S * 31 + X) mod 1000000007), A, 0).

:- pred is_sorted(array(T)::in) is semidet.
is_sorted(A) :-
    not (
        list.member(I, 1 .. array.size(A) - 1),
        compare((<), A ^ elem(I), A ^ elem(I - 1))
    ).

:- func descending(int, int) = comparison_result.
descending(X, Y) = R :- compare(R, Y, X).

:- pred searches(array(int)::in, comparison_func(int)::in, list(int)::in,
    io::di, io::uo) is det.
searches(A, Cmp, Xs, !IO) :-
    list.map(
        (pred(X::in, S::out) is det :-
            ( if array.binary_search(Cmp, A, X, I) then
                BS = string.int_to_string(I)
            else
                BS = "no"
            ),
            ( if array.approx_binary_search(Cmp, A, X, J) then
                AS = string.int_to_string(J)
            else
                AS = "no"
            ),
            S = string.format("%d:%s/%s", [i(X), s(BS), s(AS)])
        ), Xs, Ss),
    io.write_string(string.join_list(" ", Ss), !IO),
    io.nl(!IO).

main(!IO) :-
    % Sorting.
    list.foldl(
        (pred(N::in, !.IO::di, !:IO::uo) is det :-
            A = array.sort(array.from_list(numbers(N, 50, N + 7))),
            io.format("sort %d: %d %s\n", [i(N), i(checksum(A)),
                s(( if is_sorted(A) then "sorted" else "unsorted" ))], !IO)
        ), [0, 1, 2, 3, 7, 8, 9, 64, 100, 1000], !IO),
    io.write_line(array.sort(array(1 .. 10)), !IO),
    io.write_line(array.sort(array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])), !IO),
    io.write_line(array.sort(array([3, 3, 3, 1, 1, 2])), !IO),
    io.write_line(array.sort(array(["pear", "apple", "fig", "apple", ""])),
        !IO),
    io.write_line(array.sort(array([2.5, -1.0, 0.0, 2.5, -7.25])), !IO),
    io.write_line(array.sort(array([3i8, -128i8, 127i8, 0i8, -1i8])), !IO),
    io.write_line(array.sort(array(['z', 'a', 'é', 'A'])), !IO),
    io.write_line(array.sort(array([rect(2, 1), circle(1.5), rect(1, 9),
        circle(-1.0)])), !IO),
    array.sort_fix_2014,
    % Folding from the right.
    A1 = array([1, 2, 3, 4]),
    io.write_line(array.foldr((func(X, S) = S * 10 + X), A1, 0), !IO),
    io.write_line(array.foldr((func(X, S) = [X | S]), A1, []), !IO),
    array.foldr((pred(X::in, S0::in, S::out) is det :- S = S0 ++ [X]),
        A1, [], L1),
    io.write_line(L1, !IO),
    array.foldr((pred(X::in, !.IO::di, !:IO::uo) is det :-
        io.write_int(X, !IO)), A1, !IO),
    io.nl(!IO),
    ( if array.foldr((pred(X::in, S0::in, S::out) is semidet :-
            X > 0, S = S0 + X), A1, 0, S1) then
        io.write_line(S1, !IO)
    else
        io.write_string("foldr failed\n", !IO)
    ),
    ( if array.foldr((pred(X::in, S0::in, S::out) is semidet :-
            X < 4, S = S0 + X), A1, 0, S2) then
        io.write_line(S2, !IO)
    else
        io.write_string("foldr failed at the first\n", !IO)
    ),
    array.foldr2((pred(X::in, P0::in, P::out, Q0::in, Q::out) is det :-
        P = P0 * 10 + X, Q = Q0 + X), A1, 0, P2, 0, Q2),
    io.write_line({P2, Q2}, !IO),
    array.foldr3((pred(X::in, P0::in, P::out, Q0::in, Q::out,
            !.IO::di, !:IO::uo) is det :-
        P = P0 * 10 + X, Q = Q0 + X, io.write_int(X, !IO)),
        A1, 0, P3, 0, Q3, !IO),
    io.write_line({P3, Q3}, !IO),
    array.foldr5((pred(X::in, B0::in, B::out, C0::in, C::out, D0::in, D::out,
            E0::in, E::out, F0::in, F::out) is det :-
        B = B0 + X, C = C0 * X, D = [X | D0], E = E0 ++ string.int_to_string(X),
        F = int.max(F0, X)), A1, 0, B5, 1, C5, [], D5, "", E5, 0, F5),
    io.write_line({B5, C5, D5, E5, F5}, !IO),
    io.write_line(array.foldr((func(X, S) = S + X), array([]) : array(int), 7),
        !IO),
    io.write_line(array.foldr((func(S, T) = T ++ S), array(["a", "b", "c"]),
        ""), !IO),
    % Searching.
    Dups = array([1, 3, 3, 3, 5, 7, 7, 9, 9, 9, 9, 11]),
    searches(Dups, ordering, -1 .. 12, !IO),
    searches(array([]), ordering, [0, 1], !IO),
    searches(array([4]), ordering, [3, 4, 5], !IO),
    searches(array([2, 2, 2, 2, 2, 2, 2]), ordering, [1, 2, 3], !IO),
    searches(array([9, 7, 7, 5, 3, 3, 1]), descending, -1 .. 10, !IO),
    Big = array.sort(array.from_list(numbers(200, 60, 3))),
    searches(Big, ordering, 0 .. 60, !IO),
    Strs = array(["a", "c", "c", "e"]),
    list.foldl(
        (pred(S::in, !.IO::di, !:IO::uo) is det :-
            ( if array.binary_search(Strs, S, I) then
                io.format("%s:%d ", [s(S), i(I)], !IO)
            else
                io.format("%s:no ", [s(S)], !IO)
            ),
            ( if array.approx_binary_search(Strs, S, J) then
                io.format("~%d\n", [i(J)], !IO)
            else
                io.write_string("~no\n", !IO)
            )
        ), ["", "a", "b", "c", "d", "e", "f"], !IO),
    % Appending.
    io.write_line(array.append(array([1, 2]), array([3, 4, 5])), !IO),
    io.write_line(array.append(array([]), array([3])), !IO),
    io.write_line(array.append(array([1]), array([])), !IO),
    io.write_line(array.append(array([]), array([])) : array(int), !IO),
    io.write_line(array.append(array([circle(1.0)]), array([rect(1, 2)])),
        !IO),
    io.write_line(array.append(array([1.5]), array([2.5, 3.5])), !IO),
    io.write_line(array.append(array([1i8]), array([-2i8])), !IO),
    App = array.append(array.from_list(numbers(500, 1000, 1)),
        array.from_list(numbers(700, 1000, 2))),
    array.copy(App, ToSort),
    io.write_line({array.size(App) : int, checksum(App),
        checksum(array.sort(ToSort))}, !IO).
