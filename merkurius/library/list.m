% The standard library module list, with the declarations of the library
% reference manual (22.01), less those that need modules not stubbed yet
% and the less common ones (README.md), all implemented in Mercury.

:- module list.
:- interface.

:- type list(T)
    --->    []
    ;       [T | list(T)].
:- inst list_skel(I) for list/1
    --->    []
    ;       [I | list_skel(I)].
:- inst list(I) == list_skel(I).
:- inst empty_list for list/1
    --->    [].
:- inst non_empty_list for list/1
    --->    [ground | ground].
:- pred is_empty(list(T)::in) is semidet.
:- pred is_not_empty(list(T)::in) is semidet.
:- func head(list(T)) = T is semidet.
:- func det_head(list(T)) = T.
:- func tail(list(T)) = list(T) is semidet.
:- func det_tail(list(T)) = list(T).
:- func cons(T, list(T)) = list(T).
:- pred cons(T::in, list(T)::in, list(T)::out) is det.
:- pred append(list(T), list(T), list(T)).
:- mode append(di, di, uo) is det.
:- mode append(in, in, out) is det.
:- mode append(in, in, in) is semidet.
:- mode append(in, out, in) is semidet.
:- mode append(out, out, in) is multi.
:- func append(list(T), list(T)) = list(T).
:- func list(T) ++ list(T) = list(T).
:- func length(list(T)) = int.
:- pred length(list(_T), int).
:- mode length(in, out) is det.
:- pred same_length(list(T1), list(T2)).
:- mode same_length(in, in) is semidet.
:- pred member(T, list(T)).
:- mode member(in, in) is semidet.
:- mode member(out, in) is nondet.
:- pred member(T::out, list(T)::in, list(T)::out) is nondet.
:- pred contains(list(T)::in, T::in) is semidet.
:- pred index0(list(T)::in, int::in, T::out) is semidet.
:- pred index1(list(T)::in, int::in, T::out) is semidet.
:- func det_index0(list(T), int) = T.
:- pred det_index0(list(T)::in, int::in, T::out) is det.
:- func det_index1(list(T), int) = T.
:- pred det_index1(list(T)::in, int::in, T::out) is det.
:- pred nth_member_search(list(T)::in, T::in, int::out) is semidet.
:- pred reverse(list(T), list(T)).
:- mode reverse(in, out) is det.
:- mode reverse(out, in) is det.
:- func reverse(list(T)) = list(T).
:- pred insert(T, list(T), list(T)).
:- mode insert(in, in, in) is semidet.
:- mode insert(in, out, in) is nondet.
:- mode insert(out, out, in) is nondet.
:- mode insert(in, in, out) is multi.
:- pred delete(list(T), T, list(T)).
:- mode delete(in, in, in) is semidet.
:- mode delete(in, in, out) is nondet.
:- mode delete(in, out, out) is nondet.
:- mode delete(out, in, in) is multi.
:- func delete_all(list(T), T) = list(T).
:- pred delete_all(list(T), T, list(T)).
:- mode delete_all(di, in, uo) is det.
:- mode delete_all(in, in, out) is det.
:- pred replace(list(T), T, T, list(T)).
:- mode replace(in, in, in, in) is semidet.
:- mode replace(in, in, in, out) is nondet.
:- func series(T, pred(T), func(T) = T) = list(T).
:- mode series(in, pred(in) is semidet, func(in) = out is det) = out is det.
:- func remove_dups(list(T)) = list(T).
:- pred remove_dups(list(T)::in, list(T)::out) is det.
:- func merge(list(T), list(T)) = list(T).
:- pred merge(list(T)::in, list(T)::in, list(T)::out) is det.
:- func merge(comparison_func(X), list(X), list(X)) = list(X).
:- pred merge(comparison_pred(X)::in(comparison_pred),
    list(X)::in, list(X)::in, list(X)::out) is det.
:- func sort(list(T)) = list(T).
:- pred sort(list(T)::in, list(T)::out) is det.
:- func sort_and_remove_dups(list(T)) = list(T).
:- pred sort_and_remove_dups(list(T)::in, list(T)::out) is det.
:- func sort(comparison_func(X), list(X)) = list(X).
:- pred sort(comparison_pred(X)::in(comparison_pred), list(X)::in,
    list(X)::out) is det.
:- pred sort_and_remove_dups(comparison_pred(X)::in(comparison_pred),
    list(X)::in, list(X)::out) is det.
:- pred split_list(int::in, list(T)::in, list(T)::out, list(T)::out)
    is semidet.
:- pred last(list(T)::in, T::out) is semidet.
:- func det_last(list(T)) = T.
:- pred det_last(list(T)::in, T::out) is det.
:- pred take(int::in, list(T)::in, list(T)::out) is semidet.
:- func take_upto(int, list(T)) = list(T).
:- pred take_upto(int::in, list(T)::in, list(T)::out) is det.
:- pred drop(int::in, list(T)::in, list(T)::out) is semidet.
:- func duplicate(int, T) = list(T).
:- pred duplicate(int::in, T::in, list(T)::out) is det.
:- func condense(list(list(T))) = list(T).
:- pred condense(list(list(T))::in, list(T)::out) is det.
:- func zip(list(T), list(T)) = list(T).
:- pred zip(list(T)::in, list(T)::in, list(T)::out) is det.
:- pred find_first_match(pred(X)::in(pred(in) is semidet), list(X)::in,
    X::out) is semidet.
:- pred any_true(pred(X)::in(pred(in) is semidet), list(X)::in) is semidet.
:- pred all_true(pred(X)::in(pred(in) is semidet), list(X)::in) is semidet.
:- func filter(pred(X)::in(pred(in) is semidet), list(X)::in)
    = (list(X)::out) is det.
:- pred filter(pred(X)::in(pred(in) is semidet), list(X)::in,
    list(X)::out) is det.
:- pred filter(pred(X)::in(pred(in) is semidet), list(X)::in,
    list(X)::out, list(X)::out) is det.
:- func filter_map(func(X) = Y, list(X)) = list(Y).
:- mode filter_map(func(in) = out is semidet, in) = out is det.
:- pred filter_map(pred(X, Y)::in(pred(in, out) is semidet),
    list(X)::in, list(Y)::out) is det.
:- pred filter_map(pred(X, Y)::in(pred(in, out) is semidet),
    list(X)::in, list(Y)::out, list(X)::out) is det.
:- func map(func(X) = Y, list(X)) = list(Y).
:- pred map(pred(X, Y), list(X), list(Y)).
:- mode map(pred(in, out) is det, in, out) is det.
:- mode map(pred(in, out) is cc_multi, in, out) is cc_multi.
:- mode map(pred(in, out) is semidet, in, out) is semidet.
:- mode map(pred(in, out) is multi, in, out) is multi.
:- mode map(pred(in, out) is nondet, in, out) is nondet.
:- mode map(pred(in, in) is semidet, in, in) is semidet.
:- pred map2(pred(A, B, C), list(A), list(B), list(C)).
:- mode map2(pred(in, out, out) is det, in, out, out) is det.
:- mode map2(pred(in, out, out) is cc_multi, in, out, out) is cc_multi.
:- mode map2(pred(in, out, out) is semidet, in, out, out) is semidet.
:- mode map2(pred(in, out, out) is multi, in, out, out) is multi.
:- mode map2(pred(in, out, out) is nondet, in, out, out) is nondet.
:- mode map2(pred(in, in, in) is semidet, in, in, in) is semidet.
:- pred map3(pred(A, B, C, D), list(A), list(B), list(C), list(D)).
:- mode map3(pred(in, out, out, out) is det, in, out, out, out) is det.
:- mode map3(pred(in, out, out, out) is cc_multi, in, out, out, out)
    is cc_multi.
:- mode map3(pred(in, out, out, out) is semidet, in, out, out, out)
    is semidet.
:- mode map3(pred(in, out, out, out) is multi, in, out, out, out)
    is multi.
:- mode map3(pred(in, out, out, out) is nondet, in, out, out, out)
    is nondet.
:- mode map3(pred(in, in, in, in) is semidet, in, in, in, in) is semidet.
:- func foldl(func(L, A) = A, list(L), A) = A.
:- pred foldl(pred(L, A, A), list(L), A, A).
:- mode foldl(pred(in, in, out) is det, in, in, out) is det.
:- mode foldl(pred(in, mdi, muo) is det, in, mdi, muo) is det.
:- mode foldl(pred(in, di, uo) is det, in, di, uo) is det.
:- mode foldl(pred(in, in, out) is semidet, in, in, out) is semidet.
:- mode foldl(pred(in, mdi, muo) is semidet, in, mdi, muo) is semidet.
:- mode foldl(pred(in, di, uo) is semidet, in, di, uo) is semidet.
:- mode foldl(pred(in, in, out) is multi, in, in, out) is multi.
:- mode foldl(pred(in, in, out) is nondet, in, in, out) is nondet.
:- mode foldl(pred(in, mdi, muo) is nondet, in, mdi, muo) is nondet.
:- mode foldl(pred(in, in, out) is cc_multi, in, in, out) is cc_multi.
:- mode foldl(pred(in, di, uo) is cc_multi, in, di, uo) is cc_multi.
:- pred foldl2(pred(L, A, A, B, B), list(L), A, A, B, B).
:- mode foldl2(pred(in, in, out, in, out) is det,
    in, in, out, in, out) is det.
:- mode foldl2(pred(in, in, out, mdi, muo) is det,
    in, in, out, mdi, muo) is det.
:- mode foldl2(pred(in, in, out, di, uo) is det,
    in, in, out, di, uo) is det.
:- mode foldl2(pred(in, di, uo, di, uo) is det,
    in, di, uo, di, uo) is det.
:- mode foldl2(pred(in, in, out, in, out) is semidet,
    in, in, out, in, out) is semidet.
:- mode foldl2(pred(in, in, out, mdi, muo) is semidet,
    in, in, out, mdi, muo) is semidet.
:- mode foldl2(pred(in, in, out, di, uo) is semidet,
    in, in, out, di, uo) is semidet.
:- mode foldl2(pred(in, in, out, in, out) is nondet,
    in, in, out, in, out) is nondet.
:- mode foldl2(pred(in, in, out, mdi, muo) is nondet,
    in, in, out, mdi, muo) is nondet.
:- mode foldl2(pred(in, in, out, in, out) is cc_multi,
    in, in, out, in, out) is cc_multi.
:- mode foldl2(pred(in, in, out, mdi, muo) is cc_multi,
    in, in, out, mdi, muo) is cc_multi.
:- mode foldl2(pred(in, in, out, di, uo) is cc_multi,
    in, in, out, di, uo) is cc_multi.
:- mode foldl2(pred(in, di, uo, di, uo) is cc_multi,
    in, di, uo, di, uo) is cc_multi.
:- pred foldl3(pred(L, A, A, B, B, C, C), list(L),
    A, A, B, B, C, C).
:- mode foldl3(pred(in, in, out, in, out, in, out) is det,
    in, in, out, in, out, in, out) is det.
:- mode foldl3(pred(in, in, out, in, out, mdi, muo) is det,
    in, in, out, in, out, mdi, muo) is det.
:- mode foldl3(pred(in, in, out, in, out, di, uo) is det,
    in, in, out, in, out, di, uo) is det.
:- mode foldl3(pred(in, in, out, in, out, in, out) is semidet,
    in, in, out, in, out, in, out) is semidet.
:- mode foldl3(pred(in, in, out, in, out, mdi, muo) is semidet,
    in, in, out, in, out, mdi, muo) is semidet.
:- mode foldl3(pred(in, in, out, in, out, di, uo) is semidet,
    in, in, out, in, out, di, uo) is semidet.
:- mode foldl3(pred(in, in, out, in, out, in, out) is nondet,
    in, in, out, in, out, in, out) is nondet.
:- mode foldl3(pred(in, in, out, in, out, mdi, muo) is nondet,
    in, in, out, in, out, mdi, muo) is nondet.
:- mode foldl3(pred(in, in, out, in, out, in, out) is cc_multi,
    in, in, out, in, out, in, out) is cc_multi.
:- mode foldl3(pred(in, in, out, in, out, di, uo) is cc_multi,
    in, in, out, in, out, di, uo) is cc_multi.
:- func foldr(func(L, A) = A, list(L), A) = A.
:- pred foldr(pred(L, A, A), list(L), A, A).
:- mode foldr(pred(in, in, out) is det, in, in, out) is det.
:- mode foldr(pred(in, mdi, muo) is det, in, mdi, muo) is det.
:- mode foldr(pred(in, di, uo) is det, in, di, uo) is det.
:- mode foldr(pred(in, in, out) is semidet, in, in, out) is semidet.
:- mode foldr(pred(in, mdi, muo) is semidet, in, mdi, muo) is semidet.
:- mode foldr(pred(in, di, uo) is semidet, in, di, uo) is semidet.
:- mode foldr(pred(in, in, out) is multi, in, in, out) is multi.
:- mode foldr(pred(in, in, out) is nondet, in, in, out) is nondet.
:- mode foldr(pred(in, mdi, muo) is nondet, in, mdi, muo) is nondet.
:- mode foldr(pred(in, di, uo) is cc_multi, in, di, uo) is cc_multi.
:- mode foldr(pred(in, in, out) is cc_multi, in, in, out) is cc_multi.
:- pred map_foldl(pred(L, M, A, A), list(L), list(M), A, A).
:- mode map_foldl(pred(in, out, in, out) is det, in, out, in, out)
    is det.
:- mode map_foldl(pred(in, out, mdi, muo) is det, in, out, mdi, muo)
    is det.
:- mode map_foldl(pred(in, out, di, uo) is det, in, out, di, uo)
    is det.
:- mode map_foldl(pred(in, out, in, out) is semidet, in, out, in, out)
    is semidet.
:- mode map_foldl(pred(in, out, mdi, muo) is semidet, in, out, mdi, muo)
    is semidet.
:- mode map_foldl(pred(in, out, di, uo) is semidet, in, out, di, uo)
    is semidet.
:- mode map_foldl(pred(in, in, di, uo) is semidet, in, in, di, uo)
    is semidet.
:- mode map_foldl(pred(in, out, in, out) is nondet, in, out, in, out)
    is nondet.
:- mode map_foldl(pred(in, out, mdi, muo) is nondet, in, out, mdi, muo)
    is nondet.
:- mode map_foldl(pred(in, out, in, out) is cc_multi, in, out, in, out)
    is cc_multi.
:- mode map_foldl(pred(in, out, mdi, muo) is cc_multi, in, out, mdi, muo)
    is cc_multi.
:- mode map_foldl(pred(in, out, di, uo) is cc_multi, in, out, di, uo)
    is cc_multi.

:- implementation.
:- import_module int, require.

is_empty([]).

is_not_empty([_ | _]).

head(L) = H :-
    L = [H | _].

det_head(L) = H :-
    (
        L = [H | _]
    ;
        L = [],
        unexpected("function `list.det_head'/1", "empty list")
    ).

tail(L) = T :-
    L = [_ | T].

det_tail(L) = T :-
    (
        L = [_ | T]
    ;
        L = [],
        unexpected("function `list.det_tail'/1", "empty list")
    ).

cons(H, T) = [H | T].

cons(H, T, [H | T]).

append([], Ys, Ys).
append([X | Xs], Ys, [X | Zs]) :-
    append(Xs, Ys, Zs).

append(Xs, Ys) = Zs :-
    append(Xs, Ys, Zs).

Xs ++ Ys = Zs :-
    append(Xs, Ys, Zs).

length(L) = N :-
    length(L, N).

length(L, N) :-
    length_acc(L, 0, N).

:- pred length_acc(list(T)::in, int::in, int::out) is det.

length_acc([], N, N).
length_acc([_ | Xs], N0, N) :-
    length_acc(Xs, N0 + 1, N).

same_length([], []).
same_length([_ | Xs], [_ | Ys]) :-
    same_length(Xs, Ys).

% With no outputs, member(in, in) has at most one solution.
member(X, [X | _]).
member(X, [_ | Xs]) :-
    member(X, Xs).

% The suffix starting at the element.
member(X, [X | Xs], [X | Xs]).
member(X, [_ | Xs], Suffix) :-
    member(X, Xs, Suffix).

contains(L, X) :-
    member(X, L).

% A negative index fails.
index0([X0 | Xs], N, X) :-
    ( if N = 0 then
        X = X0
    else
        N > 0,
        index0(Xs, N - 1, X)
    ).

index1(L, N, X) :-
    index0(L, N - 1, X).

det_index0(L, N) = X :-
    det_index0(L, N, X).

det_index0(L, N, X) :-
    ( if index0(L, N, X0) then
        X = X0
    else
        unexpected("predicate `list.det_index0'/3", "index out of range")
    ).

det_index1(L, N) = X :-
    det_index1(L, N, X).

% As mmc, which reports an index out of range as det_index0's.
det_index1(L, N, X) :-
    det_index0(L, N - 1, X).

nth_member_search(L, X, N) :-
    nth_member_search_from(L, X, 1, N).

:- pred nth_member_search_from(list(T)::in, T::in, int::in, int::out)
    is semidet.

nth_member_search_from([X0 | Xs], X, N0, N) :-
    ( if X0 = X then
        N = N0
    else
        nth_member_search_from(Xs, X, N0 + 1, N)
    ).

:- pragma promise_equivalent_clauses(reverse/2).
reverse(L::in, R::out) :-
    reverse_acc(L, [], R).
reverse(L::out, R::in) :-
    reverse_acc(R, [], L).

reverse(L) = R :-
    reverse(L, R).

:- pred reverse_acc(list(T)::in, list(T)::in, list(T)::out) is det.

reverse_acc([], R, R).
reverse_acc([X | Xs], R0, R) :-
    reverse_acc(Xs, [X | R0], R).

insert(X, L0, L) :-
    delete(L, X, L0).

delete([X | Xs], X, Xs).
delete([X | Xs], Y, [X | Zs]) :-
    delete(Xs, Y, Zs).

delete_all(L0, X) = L :-
    delete_all(L0, X, L).

delete_all([], _, []).
delete_all([X0 | Xs], X, L) :-
    ( if X0 = X then
        delete_all(Xs, X, L)
    else
        delete_all(Xs, X, L1),
        L = [X0 | L1]
    ).

replace([X | Xs], X, Y, [Y | Xs]).
replace([X | Xs], Z, Y, [X | Zs]) :-
    replace(Xs, Z, Y, Zs).

series(I, OK, Succ) =
    ( if OK(I) then
        [I | series(Succ(I), OK, Succ)]
    else
        []
    ).

remove_dups(L0) = L :-
    remove_dups(L0, L).

% The first of each value, in order: the elements, numbered, are sorted
% by value and then position, the first of each value kept, and those put
% back in their order.
remove_dups(L0, L) :-
    number(L0, 0, Numbered),
    sort(Numbered, ByValue),
    first_of_each(ByValue, Firsts),
    sort(Firsts, ByPosition),
    strip(ByPosition, L).

:- pred number(list(T)::in, int::in, list({T, int})::out) is det.

number([], _, []).
number([X | Xs], N, [{X, N} | Ys]) :-
    number(Xs, N + 1, Ys).

:- pred first_of_each(list({T, int})::in, list({int, T})::out) is det.

first_of_each([], []).
first_of_each([{X, N} | Xs], [{N, X} | Ys]) :-
    skip_value(X, Xs, Rest),
    first_of_each(Rest, Ys).

:- pred skip_value(T::in, list({T, int})::in, list({T, int})::out) is det.

skip_value(_, [], []).
skip_value(X, [{Y, N} | Ys], Rest) :-
    ( if X = Y then
        skip_value(X, Ys, Rest)
    else
        Rest = [{Y, N} | Ys]
    ).

:- pred strip(list({int, T})::in, list(T)::out) is det.

strip([], []).
strip([{_, X} | Xs], [X | Ys]) :-
    strip(Xs, Ys).

merge(L1, L2) = L :-
    merge(L1, L2, L).

% On a tie, the first list's element first.
merge([], L2, L2).
merge([X | Xs], L2, L) :-
    (
        L2 = [],
        L = [X | Xs]
    ;
        L2 = [Y | Ys],
        compare(R, X, Y),
        ( if R = (>) then
            merge([X | Xs], Ys, L1),
            L = [Y | L1]
        else
            merge(Xs, [Y | Ys], L1),
            L = [X | L1]
        )
    ).

merge(F, L1, L2) = L :-
    merge(function_pred(F), L1, L2, L).

merge(_, [], L2, L2).
merge(P, [X | Xs], L2, L) :-
    (
        L2 = [],
        L = [X | Xs]
    ;
        L2 = [Y | Ys],
        P(X, Y, R),
        ( if R = (>) then
            merge(P, [X | Xs], Ys, L1),
            L = [Y | L1]
        else
            merge(P, Xs, [Y | Ys], L1),
            L = [X | L1]
        )
    ).

:- func function_pred(comparison_func(T)) = comparison_pred(T).
:- mode function_pred(in(comparison_func)) = out(comparison_pred) is det.

function_pred(F) = (pred(X::in, Y::in, R::out) is det :- R = F(X, Y)).

sort(L0) = L :-
    sort(L0, L).

% A merge sort, which is stable.
sort(L0, L) :-
    length(L0, N),
    sort_prefix(N, L0, L, _).

% The first N elements of L0 sorted, and the rest.
:- pred sort_prefix(int::in, list(T)::in, list(T)::out, list(T)::out)
    is det.

sort_prefix(N, L0, Sorted, Rest) :-
    ( if N > 1 then
        Half = N / 2,
        sort_prefix(Half, L0, A, L1),
        sort_prefix(N - Half, L1, B, Rest),
        merge(A, B, Sorted)
    else if N = 1, L0 = [X | Xs] then
        Sorted = [X],
        Rest = Xs
    else
        Sorted = [],
        Rest = L0
    ).

sort_and_remove_dups(L0) = L :-
    sort_and_remove_dups(L0, L).

sort_and_remove_dups(L0, L) :-
    sort(L0, L1),
    remove_adjacent_dups(L1, L).

:- pred remove_adjacent_dups(list(T)::in, list(T)::out) is det.

remove_adjacent_dups([], []).
remove_adjacent_dups([X | Xs], [X | Ys]) :-
    remove_adjacent_dups_after(X, Xs, Ys).

:- pred remove_adjacent_dups_after(T::in, list(T)::in, list(T)::out)
    is det.

remove_adjacent_dups_after(_, [], []).
remove_adjacent_dups_after(X, [Y | Ys], Zs) :-
    ( if X = Y then
        remove_adjacent_dups_after(X, Ys, Zs)
    else
        Zs = [Y | Zs1],
        remove_adjacent_dups_after(Y, Ys, Zs1)
    ).

sort(F, L0) = L :-
    sort(function_pred(F), L0, L).

sort(P, L0, L) :-
    length(L0, N),
    sort_prefix(P, N, L0, L, _).

:- pred sort_prefix(comparison_pred(T)::in(comparison_pred), int::in,
    list(T)::in, list(T)::out, list(T)::out) is det.

sort_prefix(P, N, L0, Sorted, Rest) :-
    ( if N > 1 then
        Half = N / 2,
        sort_prefix(P, Half, L0, A, L1),
        sort_prefix(P, N - Half, L1, B, Rest),
        merge(P, A, B, Sorted)
    else if N = 1, L0 = [X | Xs] then
        Sorted = [X],
        Rest = Xs
    else
        Sorted = [],
        Rest = L0
    ).

% The first of the elements that compare equal, as the sort is stable.
sort_and_remove_dups(P, L0, L) :-
    sort(P, L0, L1),
    remove_adjacent_dups(P, L1, L).

:- pred remove_adjacent_dups(comparison_pred(T)::in(comparison_pred),
    list(T)::in, list(T)::out) is det.

remove_adjacent_dups(_, [], []).
remove_adjacent_dups(P, [X | Xs], [X | Ys]) :-
    remove_adjacent_dups_after(P, X, Xs, Ys).

:- pred remove_adjacent_dups_after(comparison_pred(T)::in(comparison_pred),
    T::in, list(T)::in, list(T)::out) is det.

remove_adjacent_dups_after(_, _, [], []).
remove_adjacent_dups_after(P, X, [Y | Ys], Zs) :-
    P(X, Y, R),
    ( if R = (=) then
        remove_adjacent_dups_after(P, X, Ys, Zs)
    else
        Zs = [Y | Zs1],
        remove_adjacent_dups_after(P, Y, Ys, Zs1)
    ).

% A negative count fails, as does one beyond the list's length.
split_list(N, L, Start, End) :-
    ( if N = 0 then
        Start = [],
        End = L
    else
        N > 0,
        L = [X | Xs],
        Start = [X | Start1],
        split_list(N - 1, Xs, Start1, End)
    ).

last([X | Xs], Last) :-
    (
        Xs = [],
        Last = X
    ;
        Xs = [_ | _],
        last(Xs, Last)
    ).

det_last(L) = X :-
    det_last(L, X).

det_last(L, X) :-
    ( if last(L, X0) then
        X = X0
    else
        unexpected("predicate `list.det_last'/2", "empty list")
    ).

take(N, L, Start) :-
    ( if N = 0 then
        Start = []
    else
        N > 0,
        L = [X | Xs],
        Start = [X | Start1],
        take(N - 1, Xs, Start1)
    ).

take_upto(N, L) = Start :-
    take_upto(N, L, Start).

take_upto(N, L, Start) :-
    ( if N < 0 then
        unexpected("list.m", "predicate `list.take_upto'/3",
            "index is negative")
    else
        take_upto_count(N, L, Start)
    ).

:- pred take_upto_count(int::in, list(T)::in, list(T)::out) is det.

take_upto_count(N, L, Start) :-
    ( if N > 0, L = [X | Xs] then
        Start = [X | Start1],
        take_upto_count(N - 1, Xs, Start1)
    else
        Start = []
    ).

drop(N, L, End) :-
    ( if N = 0 then
        End = L
    else
        N > 0,
        L = [_ | Xs],
        drop(N - 1, Xs, End)
    ).

duplicate(N, X) = L :-
    duplicate(N, X, L).

% A count that is not positive gives the empty list.
duplicate(N, X, L) :-
    ( if N > 0 then
        L = [X | L1],
        duplicate(N - 1, X, L1)
    else
        L = []
    ).

condense(Ls) = L :-
    condense(Ls, L).

condense([], []).
condense([L | Ls], R) :-
    condense(Ls, R1),
    append(L, R1, R).

% The elements alternately, and the rest of the longer list.
zip(Xs, Ys) = Zs :-
    zip(Xs, Ys, Zs).

zip([], Ys, Ys).
zip([X | Xs], Ys, [X | Zs]) :-
    zip(Ys, Xs, Zs).

find_first_match(P, [X | Xs], Match) :-
    ( if P(X) then
        Match = X
    else
        find_first_match(P, Xs, Match)
    ).

any_true(P, [X | Xs]) :-
    ( if P(X) then
        true
    else
        any_true(P, Xs)
    ).

all_true(_, []).
all_true(P, [X | Xs]) :-
    P(X),
    all_true(P, Xs).

filter(P, Xs) = Trues :-
    filter(P, Xs, Trues).

filter(_, [], []).
filter(P, [X | Xs], Trues) :-
    % The recursive calls are last, so that they are tail calls (last call
    % modulo constructor).
    ( if P(X) then
        filter(P, Xs, Trues1),
        Trues = [X | Trues1]
    else
        filter(P, Xs, Trues)
    ).

filter(_, [], [], []).
filter(P, [X | Xs], Trues, Falses) :-
    filter(P, Xs, Trues1, Falses1),
    ( if P(X) then
        Trues = [X | Trues1],
        Falses = Falses1
    else
        Trues = Trues1,
        Falses = [X | Falses1]
    ).

filter_map(_, []) = [].
filter_map(F, [X | Xs]) = Ys :-
    % The recursive calls are last, as in filter/3.
    ( if Y = F(X) then
        Ys = [Y | filter_map(F, Xs)]
    else
        Ys = filter_map(F, Xs)
    ).

filter_map(_, [], []).
filter_map(P, [X | Xs], Ys) :-
    % The recursive calls are last, as in filter/3.
    ( if P(X, Y) then
        filter_map(P, Xs, Ys1),
        Ys = [Y | Ys1]
    else
        filter_map(P, Xs, Ys)
    ).

filter_map(_, [], [], []).
filter_map(P, [X | Xs], Ys, Falses) :-
    filter_map(P, Xs, Ys1, Falses1),
    ( if P(X, Y) then
        Ys = [Y | Ys1],
        Falses = Falses1
    else
        Ys = Ys1,
        Falses = [X | Falses1]
    ).

map(_, []) = [].
map(F, [X | Xs]) = [F(X) | map(F, Xs)].

map(_, [], []).
map(P, [X | Xs], [Y | Ys]) :-
    P(X, Y),
    map(P, Xs, Ys).

map2(_, [], [], []).
map2(P, [A | As], [B | Bs], [C | Cs]) :-
    P(A, B, C),
    map2(P, As, Bs, Cs).

map3(_, [], [], [], []).
map3(P, [A | As], [B | Bs], [C | Cs], [D | Ds]) :-
    P(A, B, C, D),
    map3(P, As, Bs, Cs, Ds).

foldl(_, [], A) = A.
foldl(F, [X | Xs], A) = foldl(F, Xs, F(X, A)).

foldl(_, [], !A).
foldl(P, [X | Xs], !A) :-
    P(X, !A),
    foldl(P, Xs, !A).

foldl2(_, [], !A, !B).
foldl2(P, [X | Xs], !A, !B) :-
    P(X, !A, !B),
    foldl2(P, Xs, !A, !B).

foldl3(_, [], !A, !B, !C).
foldl3(P, [X | Xs], !A, !B, !C) :-
    P(X, !A, !B, !C),
    foldl3(P, Xs, !A, !B, !C).

foldr(_, [], A) = A.
foldr(F, [X | Xs], A) = F(X, foldr(F, Xs, A)).

foldr(_, [], !A).
foldr(P, [X | Xs], !A) :-
    foldr(P, Xs, !A),
    P(X, !A).

map_foldl(_, [], [], !A).
map_foldl(P, [X | Xs], [Y | Ys], !A) :-
    P(X, Y, !A),
    map_foldl(P, Xs, Ys, !A).
