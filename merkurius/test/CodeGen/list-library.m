% The library's list module, compiled into libmerkstd: each predicate and
% function, in each mode a program can call without collecting solutions,
% the nondet and multi ones for their first solution. A call in a mode
% with its own clauses, as reverse(out, in), calls that mode's procedure.
% The output is mmc's for the same program, but for a range from min_int,
% which is the manual's: mmc's runs until memory runs out (design document,
% "Where mmc and the manual disagree").
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module list_library.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module char, int, list, pair.

:- pred show(T::in, io::di, io::uo) is det.
show(X, !IO) :- io.write(X, !IO), io.nl(!IO).

:- pred yes_no(bool_goal::in(bool_goal), io::di, io::uo) is det.
:- type bool_goal == (pred).
:- inst bool_goal == ((pred) is semidet).
yes_no(G, !IO) :-
    ( if G then io.write_string("yes\n", !IO) else io.write_string("no\n", !IO) ).

:- func ints = list(int).
ints = [1, 2, 3, 4, 5].

:- func pairs = list(pair(int, string)).
pairs = [2 - "a", 1 - "b", 2 - "c", 1 - "d", 3 - "e", 2 - "f"].

:- pred by_key(pair(int, string)::in, pair(int, string)::in,
    comparison_result::out) is det.
by_key(A - _, B - _, R) :- compare(R, A, B).

:- func by_key_func(pair(int, string), pair(int, string)) = comparison_result.
by_key_func(A - _, B - _) = R :- compare(R, A, B).

main(!IO) :-
    basics(!IO),
    searching(!IO),
    building(!IO),
    sorting(!IO),
    higher_order(!IO),
    folds(!IO),
    nondet_firsts(!IO),
    ranges(!IO).

:- pred ranges(io::di, io::uo) is det.
ranges(!IO) :-
    show(range(1, 5), !IO),
    show(range(3, 3), !IO),
    show(range(5, 1), !IO),
    show(range(-2, 2), !IO),
    show(list.length(range(1, 1000000)) : int, !IO),
    show(range(max_int - 1, max_int), !IO),
    show(range(min_int, min_int + 1), !IO).

:- func range(int, int) = list(int).
range(Lo, Hi) = Lo .. Hi.

:- pred basics(io::di, io::uo) is det.
basics(!IO) :-
    L = ints,
    yes_no((pred) is semidet :- is_empty([] : list(int)), !IO),
    yes_no((pred) is semidet :- is_not_empty(L), !IO),
    ( if H = head(L) : int then show(H, !IO) else true ),
    ( if T = tail(L) : list(int) then show(T, !IO) else true ),
    yes_no((pred) is semidet :- _ = head([] : list(int)) : int, !IO),
    show(det_head(L) : int, !IO),
    show(det_tail(L) : list(int), !IO),
    show(cons(0, L) : list(int), !IO),
    cons(0, L, C), show(C, !IO),
    show(append(L, [6]) : list(int), !IO),
    show(L ++ [6, 7], !IO),
    append([0], L, A1), show(A1, !IO),
    yes_no((pred) is semidet :- append([1, 2], [3, 4, 5], L), !IO),
    yes_no((pred) is semidet :- append([1, 2], [4], L), !IO),
    ( if append([1, 2], S, L) then show(S, !IO) else true ),
    yes_no((pred) is semidet :- append([2], _, L), !IO),
    show(length(L) : int, !IO),
    length([] : list(int), N0), show(N0, !IO),
    yes_no((pred) is semidet :- same_length(L, ["a", "b", "c", "d", "e"]), !IO),
    yes_no((pred) is semidet :- same_length(L, [1]), !IO),
    U = [1, 2] ++ [3], append(U, [4], U2), show(U2, !IO).

:- pred searching(io::di, io::uo) is det.
searching(!IO) :-
    L = ints,
    yes_no((pred) is semidet :- member(3, L), !IO),
    yes_no((pred) is semidet :- member(9, L), !IO),
    yes_no((pred) is semidet :- contains(L, 5), !IO),
    ( if index0(L, 0, I0) then show(I0, !IO) else true ),
    ( if index0(L, 4, I4) then show(I4, !IO) else true ),
    yes_no((pred) is semidet :- index0(L, 5, _), !IO),
    yes_no((pred) is semidet :- index0(L, -1, _), !IO),
    ( if index1(L, 1, J1) then show(J1, !IO) else true ),
    yes_no((pred) is semidet :- index1(L, 0, _), !IO),
    show(det_index0(L, 2) : int, !IO),
    det_index0(L, 3, D3), show(D3, !IO),
    show(det_index1(L, 2) : int, !IO),
    det_index1(L, 5, D5), show(D5, !IO),
    ( if nth_member_search([5, 6, 7, 6], 6, NM) then show(NM, !IO) else true ),
    yes_no((pred) is semidet :- nth_member_search(L, 9, _), !IO),
    ( if last(L, La) then show(La, !IO) else true ),
    yes_no((pred) is semidet :- last([] : list(int), _), !IO),
    show(det_last(L) : int, !IO),
    det_last([7], DL), show(DL, !IO),
    ( if split_list(2, L, SA, SB) then show({SA, SB}, !IO) else true ),
    ( if split_list(0, L, SC, SD) then show({SC, SD}, !IO) else true ),
    ( if split_list(5, L, SE, SF) then show({SE, SF}, !IO) else true ),
    yes_no((pred) is semidet :- split_list(6, L, _, _), !IO),
    yes_no((pred) is semidet :- split_list(-1, L, _, _), !IO),
    ( if take(3, L, TK) then show(TK, !IO) else true ),
    yes_no((pred) is semidet :- take(6, L, _), !IO),
    yes_no((pred) is semidet :- take(-1, L, _), !IO),
    show(take_upto(3, L) : list(int), !IO),
    take_upto(9, L, TU), show(TU, !IO),
    ( if drop(2, L, DR) then show(DR, !IO) else true ),
    ( if drop(5, L, DR5) then show(DR5, !IO) else true ),
    yes_no((pred) is semidet :- drop(6, L, _), !IO),
    yes_no((pred) is semidet :- drop(-1, L, _), !IO).

:- pred building(io::di, io::uo) is det.
building(!IO) :-
    L = ints,
    show(reverse(L) : list(int), !IO),
    reverse(L, R1), show(R1, !IO),
    reverse(R2, L), show(R2, !IO),
    yes_no((pred) is semidet :- insert(0, [1, 2], [1, 0, 2]), !IO),
    yes_no((pred) is semidet :- insert(0, [1, 2], [1, 2]), !IO),
    yes_no((pred) is semidet :- delete([1, 2, 3], 2, [1, 3]), !IO),
    yes_no((pred) is semidet :- delete([1, 2, 3], 4, [1, 2, 3]), !IO),
    show(delete_all([1, 2, 1, 3, 1], 1) : list(int), !IO),
    delete_all([1, 2, 1, 3, 1], 3, DA), show(DA, !IO),
    UL = [4, 5, 4] ++ [6], delete_all(UL, 4, DU), show(DU, !IO),
    yes_no((pred) is semidet :- replace([1, 2, 1], 1, 9, [1, 2, 9]), !IO),
    yes_no((pred) is semidet :- replace([1, 2, 1], 2, 9, [1, 2, 1]), !IO),
    show(series(1, (pred(X::in) is semidet :- X < 50), (func(X) = X * 3)) : list(int), !IO),
    show(remove_dups([3, 1, 3, 2, 1, 4]) : list(int), !IO),
    remove_dups([] : list(int), RD), show(RD, !IO),
    show(duplicate(3, 'x') : list(char), !IO),
    show(duplicate(-2, 'x') : list(char), !IO),
    duplicate(0, 1, DZ), show(DZ, !IO),
    show(condense([[1], [], [2, 3]]) : list(int), !IO),
    condense([] : list(list(int)), CE), show(CE, !IO),
    show(zip([1, 2, 3, 4], [10, 20]) : list(int), !IO),
    zip([1, 2], [10, 20, 30, 40], Z2), show(Z2, !IO).

:- pred sorting(io::di, io::uo) is det.
sorting(!IO) :-
    P = pairs,
    show(sort([3, 1, 2, 3, 1]) : list(int), !IO),
    sort([] : list(int), S0), show(S0, !IO),
    show(sort_and_remove_dups([3, 1, 2, 3, 1]) : list(int), !IO),
    sort_and_remove_dups(["b", "a", "b"], SR), show(SR, !IO),
    show(merge([1, 3, 5], [2, 3, 4]) : list(int), !IO),
    merge([1, 2], [] : list(int), M2), show(M2, !IO),
    show(merge(by_key_func, [1 - "a", 2 - "b", 3 - "c"], [1 - "x", 2 - "y"]) : list(pair(int, string)), !IO),
    merge(by_key, [1 - "a", 2 - "b"], [1 - "x", 2 - "y"], MP), show(MP, !IO),
    show(sort(by_key_func, P) : list(pair(int, string)), !IO),
    sort(by_key, P, SP), show(SP, !IO),
    sort_and_remove_dups(by_key, P, SRP), show(SRP, !IO),
    show(sort([5, 4, 3, 2, 1, 0, -1, 9, 8, 7, 6]) : list(int), !IO).

:- pred higher_order(io::di, io::uo) is cc_multi.
higher_order(!IO) :-
    L = ints,
    ( if find_first_match((pred(X::in) is semidet :- X > 2), L, FF) then
        show(FF, !IO)
    else
        true
    ),
    yes_no((pred) is semidet :- find_first_match((pred(X::in) is semidet :- X > 9), L, _), !IO),
    yes_no((pred) is semidet :- any_true((pred(X::in) is semidet :- X > 4), L), !IO),
    yes_no((pred) is semidet :- any_true((pred(X::in) is semidet :- X > 5), L), !IO),
    yes_no((pred) is semidet :- all_true((pred(X::in) is semidet :- X > 0), L), !IO),
    yes_no((pred) is semidet :- all_true((pred(X::in) is semidet :- X > 1), L), !IO),
    show(filter((pred(X::in) is semidet :- X mod 2 = 0), L) : list(int), !IO),
    filter((pred(X::in) is semidet :- X > 3), L, F3), show(F3, !IO),
    filter((pred(X::in) is semidet :- X mod 2 = 0), L, Ev, Od), show({Ev, Od}, !IO),
    show(filter_map((func(X::in) = (Y::out) is semidet :- X > 2, Y = X * X), L) : list(int), !IO),
    filter_map((pred(X::in, Y::out) is semidet :- X < 3, Y = X * -1), L, FM3), show(FM3, !IO),
    filter_map((pred(X::in, Y::out) is semidet :- X > 2, Y = X * X), L, FM, FMo),
    show({FM, FMo}, !IO),
    show(map((func(X) = X + 1), L) : list(int), !IO),
    map((pred(X::in, Y::out) is det :- Y = X * 2), L, M1), show(M1, !IO),
    map((pred(X::in, Y::out) is cc_multi :- (Y = X ; Y = X * -1)), L, M2), show(M2, !IO),
    ( if map((pred(X::in, Y::out) is semidet :- X < 9, Y = X), L, M3) then show(M3, !IO) else true ),
    yes_no((pred) is semidet :- map((pred(X::in, Y::out) is semidet :- X < 3, Y = X), L, _), !IO),
    yes_no((pred) is semidet :- map((pred(X::in, Y::in) is semidet :- Y = X + 1), [1, 2], [2, 3]), !IO),
    map2((pred(X::in, Y::out, Z::out) is det :- Y = X * 2, Z = X * 3), L, A2, B2),
    show({A2, B2}, !IO),
    map2((pred(X::in, Y::out, Z::out) is cc_multi :- Y = X, Z = X * -1), [1], A3, B3),
    show({A3, B3}, !IO),
    ( if map2((pred(X::in, Y::out, Z::out) is semidet :- X > 0, Y = X, Z = X), L, A4, B4) then
        show({A4, B4}, !IO)
    else
        true
    ),
    yes_no((pred) is semidet :- map2((pred(X::in, Y::in, Z::in) is semidet :- X = Y, Y = Z), [1], [1], [1]), !IO),
    map3((pred(X::in, Y::out, Z::out, W::out) is det :- Y = X, Z = X * X, W = X * X * X), [1, 2, 3], A5, B5, C5),
    show({A5, B5, C5}, !IO),
    map3((pred(X::in, Y::out, Z::out, W::out) is cc_multi :- Y = X, Z = X, W = X), [4], A6, B6, C6),
    show({A6, B6, C6}, !IO),
    ( if map3((pred(X::in, Y::out, Z::out, W::out) is semidet :- X > 0, Y = X, Z = X, W = X), [5], A7, B7, C7) then
        show({A7, B7, C7}, !IO)
    else
        true
    ),
    yes_no((pred) is semidet :- map3((pred(X::in, Y::in, Z::in, W::in) is semidet :- X = Y, Z = W), [1], [1], [2], [3]), !IO).

:- pred folds(io::di, io::uo) is cc_multi.
folds(!IO) :-
    L = ints,
    show(foldl((func(X, A) = [X | A]), L, []) : list(int), !IO),
    foldl((pred(X::in, A0::in, A::out) is det :- A = A0 * 10 + X), L, 0, F1), show(F1, !IO),
    foldl((pred(X::in, !.IO::di, !:IO::uo) is det :- io.write_int(X, !IO)), L, !IO), io.nl(!IO),
    ( if foldl((pred(X::in, A0::in, A::out) is semidet :- X < 9, A = A0 + X), L, 0, F2) then show(F2, !IO) else true ),
    yes_no((pred) is semidet :- foldl((pred(X::in, A0::in, A::out) is semidet :- X < 3, A = A0 + X), L, 0, _), !IO),
    foldl((pred(X::in, A0::in, A::out) is cc_multi :- (A = A0 + X ; A = A0)), L, 0, F3), show(F3, !IO),
    foldl((pred(X::in, !.IO::di, !:IO::uo) is cc_multi :- io.write_int(X * 2, !IO)), L, !IO), io.nl(!IO),
    foldl2((pred(X::in, A0::in, A::out, B0::in, B::out) is det :- A = A0 + X, B = B0 * X), L, 0, S, 1, Pr),
    show({S, Pr}, !IO),
    foldl2((pred(X::in, N0::in, N::out, !.IO::di, !:IO::uo) is det :- N = N0 + 1, io.write_int(X, !IO)), L, 0, Count, !IO),
    io.nl(!IO), show(Count, !IO),
    ( if foldl2((pred(X::in, A0::in, A::out, B0::in, B::out) is semidet :- X > 0, A = A0 + X, B = B0 + 1), L, 0, S2, 0, C2) then
        show({S2, C2}, !IO)
    else
        true
    ),
    foldl2((pred(X::in, A0::in, A::out, B0::in, B::out) is cc_multi :- A = A0 + X, B = B0), L, 0, S3, 0, _), show(S3, !IO),
    foldl3((pred(X::in, A0::in, A::out, B0::in, B::out, C0::in, C::out) is det :- A = A0 + X, B = B0 + 1, C = [X | C0]), L, 0, S4, 0, N4, [], R4),
    show({S4, N4, R4}, !IO),
    foldl3((pred(X::in, A0::in, A::out, B0::in, B::out, !.IO::di, !:IO::uo) is det :- A = A0 + X, B = B0 + 1, io.write_int(X, !IO)), L, 0, S5, 0, N5, !IO),
    io.nl(!IO), show({S5, N5}, !IO),
    show(foldr((func(X, A) = [X | A]), L, []) : list(int), !IO),
    foldr((pred(X::in, A0::in, A::out) is det :- A = A0 * 10 + X), L, 0, R1), show(R1, !IO),
    foldr((pred(X::in, !.IO::di, !:IO::uo) is det :- io.write_int(X, !IO)), L, !IO), io.nl(!IO),
    ( if foldr((pred(X::in, A0::in, A::out) is semidet :- X > 0, A = [X | A0]), L, [], R2) then show(R2, !IO) else true ),
    foldr((pred(X::in, A0::in, A::out) is cc_multi :- A = A0 + X), L, 0, R3), show(R3, !IO),
    map_foldl((pred(X::in, Y::out, A0::in, A::out) is det :- Y = X + A0, A = A0 + 1), L, MF, 0, MFA),
    show({MF, MFA}, !IO),
    map_foldl((pred(X::in, Y::out, !.IO::di, !:IO::uo) is det :- Y = X * X, io.write_int(X, !IO)), L, MF2, !IO),
    io.nl(!IO), show(MF2, !IO),
    ( if map_foldl((pred(X::in, Y::out, A0::in, A::out) is semidet :- X > 0, Y = X, A = A0 + X), L, MF3, 0, MFA3) then
        show({MF3, MFA3}, !IO)
    else
        true
    ),
    map_foldl((pred(X::in, Y::out, A0::in, A::out) is cc_multi :- Y = X, A = A0 + X), L, MF4, 0, MFA4),
    show({MF4, MFA4}, !IO).

% The first solution of each nondet and multi mode.
:- pred nondet_firsts(io::di, io::uo) is cc_multi.
nondet_firsts(!IO) :-
    L = ints,
    ( if append(X1, Y1, L), X1 = [_, _] then show({X1, Y1}, !IO) else true ),
    ( if member(M, L), M > 3 then show(M, !IO) else true ),
    ( if member(M3, [1, 2, 1, 3], S3), M3 = 1, S3 = [_, _] then show(S3, !IO) else true ),
    ( if insert(0, [1, 2], I1), I1 = [_, 0 | _] then show(I1, !IO) else true ),
    ( if insert(1, I2, [2, 1, 3]) then show(I2, !IO) else true ),
    ( if insert(I3, I4, [7, 8]), I3 = 8 then show(I4, !IO) else true ),
    ( if delete([1, 2, 1], 1, D1), D1 = [1, 2] then show(D1, !IO) else true ),
    ( if delete([1, 2, 3], D2, D3), D2 > 1 then show({D2, D3}, !IO) else true ),
    ( if delete(D4, 0, [1, 2]), D4 = [_, _, 0] then show(D4, !IO) else true ),
    ( if replace([1, 2, 1], 1, 9, R1), R1 = [1 | _] then show(R1, !IO) else true ),
    ( if map((pred(X::in, Y::out) is nondet :- (Y = X ; Y = X * 10)), [1, 2], MN), MN = [_, 20] then
        show(MN, !IO)
    else
        true
    ),
    map((pred(X::in, Y::out) is multi :- (Y = X ; Y = X * -1)), [1, 2], MM), show(MM, !IO),
    ( if map2((pred(X::in, Y::out, Z::out) is nondet :- (Y = X, Z = X ; Y = 0, Z = 0)), [1, 2], A2, B2), B2 = [_, 0] then
        show({A2, B2}, !IO)
    else
        true
    ),
    map2((pred(X::in, Y::out, Z::out) is multi :- (Y = X, Z = X ; Y = 0, Z = 0)), [1], A3, B3), show({A3, B3}, !IO),
    ( if map3((pred(X::in, Y::out, Z::out, W::out) is nondet :- (Y = X, Z = X, W = X ; Y = 0, Z = 0, W = 0)), [1, 2], A4, B4, C4), C4 = [0, _] then
        show({A4, B4, C4}, !IO)
    else
        true
    ),
    map3((pred(X::in, Y::out, Z::out, W::out) is multi :- (Y = X, Z = X, W = X ; Y = 0, Z = 0, W = 0)), [1], A5, B5, C5), show({A5, B5, C5}, !IO),
    ( if foldl((pred(X::in, A0::in, A::out) is nondet :- (A = A0 + X ; A = A0 * X)), [2, 3], 1, F1), F1 < 6 then show(F1, !IO) else true ),
    foldl((pred(X::in, A0::in, A::out) is multi :- (A = A0 + X ; A = A0 * X)), [2, 3], 1, F2), show(F2, !IO),
    ( if foldl2((pred(X::in, A0::in, A::out, B0::in, B::out) is nondet :- (A = A0 + X, B = B0 ; A = A0, B = B0 + X)), [2, 3], 0, F3, 0, F4), F3 = 0 then
        show({F3, F4}, !IO)
    else
        true
    ),
    ( if foldl3((pred(X::in, A0::in, A::out, B0::in, B::out, C0::in, C::out) is nondet :- (A = A0 + X, B = B0, C = C0 ; A = A0, B = B0 + X, C = C0 + 1)), [2, 3], 0, F5, 0, F6, 0, F7), F7 = 1 then
        show({F5, F6, F7}, !IO)
    else
        true
    ),
    ( if foldr((pred(X::in, A0::in, A::out) is nondet :- (A = A0 + X ; A = A0 * X)), [2, 3], 1, R3), R3 < 6 then show(R3, !IO) else true ),
    foldr((pred(X::in, A0::in, A::out) is multi :- (A = A0 + X ; A = A0 * X)), [2, 3], 1, R4), show(R4, !IO),
    ( if map_foldl((pred(X::in, Y::out, A0::in, A::out) is nondet :- (Y = X, A = A0 ; Y = 0, A = A0 + 1)), [1, 2], MF, 0, MFA), MFA = 1 then
        show({MF, MFA}, !IO)
    else
        true
    ).

% MERK-LABEL: merk.proc private @list_library.building_2_0()
% MERK: merk.call @list.reverse_f1_0(
% MERK: merk.call @list.reverse_2_0(
% MERK: merk.call @list.reverse_2_1(
% MERK-LABEL: merk.proc private @list_library.nondet_firsts_2_0()
% MERK: merk.nondet_call @list.append_3_4(
% MERK: merk.nondet_call @list.member_2_1(

% OUTPUT:yes
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:1
% OUTPUT-NEXT:[2, 3, 4, 5]
% OUTPUT-NEXT:no
% OUTPUT-NEXT:1
% OUTPUT-NEXT:[2, 3, 4, 5]
% OUTPUT-NEXT:[0, 1, 2, 3, 4, 5]
% OUTPUT-NEXT:[0, 1, 2, 3, 4, 5]
% OUTPUT-NEXT:[1, 2, 3, 4, 5, 6]
% OUTPUT-NEXT:[1, 2, 3, 4, 5, 6, 7]
% OUTPUT-NEXT:[0, 1, 2, 3, 4, 5]
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:no
% OUTPUT-NEXT:[3, 4, 5]
% OUTPUT-NEXT:no
% OUTPUT-NEXT:5
% OUTPUT-NEXT:0
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:no
% OUTPUT-NEXT:[1, 2, 3, 4]
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:no
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:1
% OUTPUT-NEXT:5
% OUTPUT-NEXT:no
% OUTPUT-NEXT:no
% OUTPUT-NEXT:1
% OUTPUT-NEXT:no
% OUTPUT-NEXT:3
% OUTPUT-NEXT:4
% OUTPUT-NEXT:2
% OUTPUT-NEXT:5
% OUTPUT-NEXT:2
% OUTPUT-NEXT:no
% OUTPUT-NEXT:5
% OUTPUT-NEXT:no
% OUTPUT-NEXT:5
% OUTPUT-NEXT:7
% OUTPUT-NEXT:{[1, 2], [3, 4, 5]}
% OUTPUT-NEXT:{[], [1, 2, 3, 4, 5]}
% OUTPUT-NEXT:{[1, 2, 3, 4, 5], []}
% OUTPUT-NEXT:no
% OUTPUT-NEXT:no
% OUTPUT-NEXT:[1, 2, 3]
% OUTPUT-NEXT:no
% OUTPUT-NEXT:no
% OUTPUT-NEXT:[1, 2, 3]
% OUTPUT-NEXT:[1, 2, 3, 4, 5]
% OUTPUT-NEXT:[3, 4, 5]
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:no
% OUTPUT-NEXT:no
% OUTPUT-NEXT:[5, 4, 3, 2, 1]
% OUTPUT-NEXT:[5, 4, 3, 2, 1]
% OUTPUT-NEXT:[5, 4, 3, 2, 1]
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:no
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:no
% OUTPUT-NEXT:[2, 3]
% OUTPUT-NEXT:[1, 2, 1, 1]
% OUTPUT-NEXT:[5, 6]
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:no
% OUTPUT-NEXT:[1, 3, 9, 27]
% OUTPUT-NEXT:[3, 1, 2, 4]
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:['x', 'x', 'x']
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:[1, 2, 3]
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:[1, 10, 2, 20, 3, 4]
% OUTPUT-NEXT:[1, 10, 2, 20, 30, 40]
% OUTPUT-NEXT:[1, 1, 2, 3, 3]
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:[1, 2, 3]
% OUTPUT-NEXT:["a", "b"]
% OUTPUT-NEXT:[1, 2, 3, 3, 4, 5]
% OUTPUT-NEXT:[1, 2]
% OUTPUT-NEXT:[1 - "a", 1 - "x", 2 - "b", 2 - "y", 3 - "c"]
% OUTPUT-NEXT:[1 - "a", 1 - "x", 2 - "b", 2 - "y"]
% OUTPUT-NEXT:[1 - "b", 1 - "d", 2 - "a", 2 - "c", 2 - "f", 3 - "e"]
% OUTPUT-NEXT:[1 - "b", 1 - "d", 2 - "a", 2 - "c", 2 - "f", 3 - "e"]
% OUTPUT-NEXT:[1 - "b", 2 - "a", 3 - "e"]
% OUTPUT-NEXT:[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
% OUTPUT-NEXT:3
% OUTPUT-NEXT:no
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:no
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:no
% OUTPUT-NEXT:[2, 4]
% OUTPUT-NEXT:[4, 5]
% OUTPUT-NEXT:{[2, 4], [1, 3, 5]}
% OUTPUT-NEXT:[9, 16, 25]
% OUTPUT-NEXT:[-1, -2]
% OUTPUT-NEXT:{[9, 16, 25], [1, 2]}
% OUTPUT-NEXT:[2, 3, 4, 5, 6]
% OUTPUT-NEXT:[2, 4, 6, 8, 10]
% OUTPUT-NEXT:[1, 2, 3, 4, 5]
% OUTPUT-NEXT:[1, 2, 3, 4, 5]
% OUTPUT-NEXT:no
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:{[2, 4, 6, 8, 10], [3, 6, 9, 12, 15]}
% OUTPUT-NEXT:{[1], [-1]}
% OUTPUT-NEXT:{[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]}
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:{[1, 2, 3], [1, 4, 9], [1, 8, 27]}
% OUTPUT-NEXT:{[4], [4], [4]}
% OUTPUT-NEXT:{[5], [5], [5]}
% OUTPUT-NEXT:no
% OUTPUT-NEXT:[5, 4, 3, 2, 1]
% OUTPUT-NEXT:12345
% OUTPUT-NEXT:12345
% OUTPUT-NEXT:15
% OUTPUT-NEXT:no
% OUTPUT-NEXT:15
% OUTPUT-NEXT:246810
% OUTPUT-NEXT:{15, 120}
% OUTPUT-NEXT:12345
% OUTPUT-NEXT:5
% OUTPUT-NEXT:{15, 5}
% OUTPUT-NEXT:15
% OUTPUT-NEXT:{15, 5, [5, 4, 3, 2, 1]}
% OUTPUT-NEXT:12345
% OUTPUT-NEXT:{15, 5}
% OUTPUT-NEXT:[1, 2, 3, 4, 5]
% OUTPUT-NEXT:54321
% OUTPUT-NEXT:54321
% OUTPUT-NEXT:[1, 2, 3, 4, 5]
% OUTPUT-NEXT:15
% OUTPUT-NEXT:{[1, 3, 5, 7, 9], 5}
% OUTPUT-NEXT:12345
% OUTPUT-NEXT:[1, 4, 9, 16, 25]
% OUTPUT-NEXT:{[1, 2, 3, 4, 5], 15}
% OUTPUT-NEXT:{[1, 2, 3, 4, 5], 15}
% OUTPUT-NEXT:{[1, 2], [3, 4, 5]}
% OUTPUT-NEXT:4
% OUTPUT-NEXT:[1, 3]
% OUTPUT-NEXT:[1, 0, 2]
% OUTPUT-NEXT:[2, 3]
% OUTPUT-NEXT:[7]
% OUTPUT-NEXT:[1, 2]
% OUTPUT-NEXT:{2, [1, 3]}
% OUTPUT-NEXT:[1, 2, 0]
% OUTPUT-NEXT:[1, 2, 9]
% OUTPUT-NEXT:[1, 20]
% OUTPUT-NEXT:[1, 2]
% OUTPUT-NEXT:{[1, 0], [1, 0]}
% OUTPUT-NEXT:{[1], [1]}
% OUTPUT-NEXT:{[0, 2], [0, 2], [0, 2]}
% OUTPUT-NEXT:{[1], [1], [1]}
% OUTPUT-NEXT:5
% OUTPUT-NEXT:6
% OUTPUT-NEXT:{0, 5}
% OUTPUT-NEXT:{2, 3, 1}
% OUTPUT-NEXT:5
% OUTPUT-NEXT:6
% OUTPUT-NEXT:{[1, 0], 1}
% OUTPUT-NEXT:[1, 2, 3, 4, 5]
% OUTPUT-NEXT:[3]
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:[-2, -1, 0, 1, 2]
% OUTPUT-NEXT:1000000
% OUTPUT-NEXT:[9223372036854775806, 9223372036854775807]
% OUTPUT-NEXT:[-9223372036854775808, -9223372036854775807]
