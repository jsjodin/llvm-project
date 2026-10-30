% A declaration-only stub of the standard library module list, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module list.
:- interface.

:- type list(T)
    --->    []
    ;       [T | list(T)].
:- inst list_skel(I) for list/1
    --->    []
    ;       [I | list_skel(I)].
:- inst list(I) == list_skel(I).
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
