% The standard library module array. The declarations are those of the
% library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md. Code generation builds in most of its
% predicates; generate, its folds and the foldl family are compiled from
% here, with mmc's messages. Its insts are the manual's work-around, in
% which arrays are not unique; the build makes sound/array.m from it, with
% unique arrays, which merkc loads unless -fmmc-uniqueness is given.

:- module array.
:- interface.

:- import_module list.

:- type array(T).

:- inst array(I) == ground.
:- inst array == array(ground).

:- inst uniq_array(I) == array(I).
:- inst uniq_array == uniq_array(ground).

:- mode array_di == di(uniq_array).
:- mode array_uo == out(uniq_array).
:- mode array_ui == in(uniq_array).

:- inst mostly_uniq_array(I) == array(I).
:- inst mostly_uniq_array == mostly_uniq_array(ground).

:- mode array_mdi == mdi(mostly_uniq_array).
:- mode array_muo == out(mostly_uniq_array).
:- mode array_mui == in(mostly_uniq_array).

:- type index_out_of_bounds
    --->    index_out_of_bounds(string).

:- pred make_empty_array(array(T)::array_uo) is det.

:- func make_empty_array = (array(T)::array_uo) is det.

:- pred init(int, T, array(T)).
:- mode init(in, in, array_uo) is det.

:- func init(int, T) = array(T).
:- mode init(in, in) = array_uo is det.

:- func array(list(T)) = array(T).
:- mode array(in) = array_uo is det.

:- func generate(int::in, (func(int) = T)::in) = (array(T)::array_uo)
    is det.

:- pred generate_foldl(int, pred(int, T, A, A), array(T), A, A).
:- mode generate_foldl(in, in(pred(in, out, in, out) is det),
    array_uo, in, out) is det.
:- mode generate_foldl(in, in(pred(in, out, mdi, muo) is det),
    array_uo, mdi, muo) is det.
:- mode generate_foldl(in, in(pred(in, out, di, uo) is det),
    array_uo, di, uo) is det.
:- mode generate_foldl(in, in(pred(in, out, in, out) is semidet),
    array_uo, in, out) is semidet.
:- mode generate_foldl(in, in(pred(in, out, mdi, muo) is semidet),
    array_uo, mdi, muo) is semidet.
:- mode generate_foldl(in, in(pred(in, out, di, uo) is semidet),
    array_uo, di, uo) is semidet.

:- pred generate_foldl2(int, pred(int, T, A, A, B, B), array(T), A, A, B, B).
:- mode generate_foldl2(in, in(pred(in, out, in, out, in, out) is det),
    array_uo, in, out, in, out) is det.
:- mode generate_foldl2(in, in(pred(in, out, mdi, muo, mdi, muo) is det),
    array_uo, mdi, muo, mdi, muo) is det.
:- mode generate_foldl2(in, in(pred(in, out, di, uo, di, uo) is det),
    array_uo, di, uo, di, uo) is det.
:- mode generate_foldl2(in, in(pred(in, out, in, out, in, out) is semidet),
    array_uo, in, out, in, out) is semidet.
:- mode generate_foldl2(in, in(pred(in, out, mdi, muo, mdi, muo) is semidet),
    array_uo, mdi, muo, mdi, muo) is semidet.
:- mode generate_foldl2(in, in(pred(in, out, di, uo, di, uo) is semidet),
    array_uo, di, uo, di, uo) is semidet.

:- pred min(array(_T), int).
:- mode min(in, out) is det.

:- func min(array(_T)) = int.
:- mode min(in) = out is det.

:- func det_least_index(array(T)) = int.
:- mode det_least_index(in) = out is det.

:- func semidet_least_index(array(T)) = int.
:- mode semidet_least_index(in) = out is semidet.

:- pred max(array(_T), int).
:- mode max(in, out) is det.

:- func max(array(_T)) = int.
:- mode max(in) = out is det.

:- func det_greatest_index(array(T)) = int.
:- mode det_greatest_index(in) = out is det.

:- func semidet_greatest_index(array(T)) = int.
:- mode semidet_greatest_index(in) = out is semidet.

:- pred size(array(_T), int).
:- mode size(in, out) is det.

:- func size(array(_T)) = int.
:- mode size(in) = out is det.

:- pred bounds(array(_T), int, int).
:- mode bounds(in, out, out) is det.

:- pred in_bounds(array(_T), int).
:- mode in_bounds(in, in) is semidet.

:- pred is_empty(array(_T)).
:- mode is_empty(in) is semidet.

:- pred lookup(array(T), int, T).
:- mode lookup(in, in, out) is det.

:- func lookup(array(T), int) = T.
:- mode lookup(in, in) = out is det.

:- pred semidet_lookup(array(T), int, T).
:- mode semidet_lookup(in, in, out) is semidet.

:- pred unsafe_lookup(array(T), int, T).
:- mode unsafe_lookup(in, in, out) is det.

:- pred set(int, T, array(T), array(T)).
:- mode set(in, in, array_di, array_uo) is det.

:- func set(array(T), int, T) = array(T).
:- mode set(array_di, in, in) = array_uo is det.

:- pred semidet_set(int, T, array(T), array(T)).
:- mode semidet_set(in, in, array_di, array_uo) is semidet.

:- pred unsafe_set(int, T, array(T), array(T)).
:- mode unsafe_set(in, in, array_di, array_uo) is det.

:- pred slow_set(int, T, array(T), array(T)).
:- mode slow_set(in, in, in, array_uo) is det.

:- func slow_set(array(T), int, T) = array(T).
:- mode slow_set(in, in, in) = array_uo is det.

:- pred semidet_slow_set(int, T, array(T), array(T)).
:- mode semidet_slow_set(in, in, in, array_uo) is semidet.

:- func elem(int, array(T)) = T.
:- mode elem(in, in) = out is det.

:- func unsafe_elem(int, array(T)) = T.
:- mode unsafe_elem(in, in) = out is det.

:- func 'elem :='(int, array(T), T) = array(T).
:- mode 'elem :='(in, array_di, in) = array_uo is det.

:- func 'unsafe_elem :='(int, array(T), T) = array(T).
:- mode 'unsafe_elem :='(in, array_di, in) = array_uo is det.

:- pred swap(int, int, array(T), array(T)).
:- mode swap(in, in, array_di, array_uo) is det.

:- pred unsafe_swap(int, int, array(T), array(T)).
:- mode unsafe_swap(in, in, array_di, array_uo) is det.

:- pred member(array(T)::in, T::out) is nondet.

:- pred copy(array(T), array(T)).
:- mode copy(in, array_uo) is det.

:- func copy(array(T)) = array(T).
:- mode copy(in) = array_uo is det.

:- pred resize(int, T, array(T), array(T)).
:- mode resize(in, in, array_di, array_uo) is det.

:- func resize(array(T), int, T) = array(T).
:- mode resize(array_di, in, in) = array_uo is det.

:- pred shrink(int, array(T), array(T)).
:- mode shrink(in, array_di, array_uo) is det.

:- func shrink(array(T), int) = array(T).
:- mode shrink(array_di, in) = array_uo is det.

:- pred fill(T::in, array(T)::array_di, array(T)::array_uo) is det.

:- pred fill_range(T::in, int::in, int::in,
     array(T)::array_di, array(T)::array_uo) is det.

:- func from_list(list(T)::in) = (array(T)::array_uo) is det.
:- pred from_list(list(T)::in, array(T)::array_uo) is det.

:- func from_reverse_list(list(T)::in) = (array(T)::array_uo) is det.

:- pred to_list(array(T), list(T)).
:- mode to_list(in, out) is det.

:- func to_list(array(T)) = list(T).
:- mode to_list(in) = out is det.

:- pred fetch_items(array(T), int, int, list(T)).
:- mode fetch_items(in, in, in, out) is det.

:- func fetch_items(array(T), int, int) = list(T).
:- mode fetch_items(in, in, in) = out is det.

:- pred binary_search(array(T)::array_ui,
    T::in, int::out) is semidet.
:- pred binary_search(comparison_func(T)::in, array(T)::array_ui,
    T::in, int::out) is semidet.

:- pred approx_binary_search(array(T)::array_ui,
    T::in, int::out) is semidet.
:- pred approx_binary_search(comparison_func(T)::in, array(T)::array_ui,
    T::in, int::out) is semidet.

:- pred map(pred(T1, T2), array(T1), array(T2)).
:- mode map(pred(in, out) is det, in, array_uo) is det.

:- func map(func(T1) = T2, array(T1)) = array(T2).
:- mode map(func(in) = out is det, in) = array_uo is det.

:- func array_compare(array(T), array(T)) = comparison_result.
:- mode array_compare(in, in) = uo is det.

:- func sort(array(T)) = array(T).
:- mode sort(array_di) = array_uo is det.

:- pred array.sort_fix_2014 is det.

:- func foldl(func(T1, T2) = T2, array(T1), T2) = T2.
:- mode foldl(func(in, in) = out is det, in, in) = out is det.
:- mode foldl(func(in, di) = uo is det, in, di) = uo is det.

:- pred foldl(pred(T1, T2, T2), array(T1), T2, T2).
:- mode foldl(pred(in, in, out) is det, in, in, out) is det.
:- mode foldl(pred(in, mdi, muo) is det, in, mdi, muo) is det.
:- mode foldl(pred(in, di, uo) is det, in, di, uo) is det.
:- mode foldl(pred(in, in, out) is semidet, in, in, out) is semidet.
:- mode foldl(pred(in, mdi, muo) is semidet, in, mdi, muo) is semidet.
:- mode foldl(pred(in, di, uo) is semidet, in, di, uo) is semidet.

:- pred foldl2(pred(T1, T2, T2, T3, T3), array(T1), T2, T2, T3, T3).
:- mode foldl2(pred(in, in, out, in, out) is det, in, in, out, in, out)
    is det.
:- mode foldl2(pred(in, in, out, mdi, muo) is det, in, in, out, mdi, muo)
    is det.
:- mode foldl2(pred(in, in, out, di, uo) is det, in, in, out, di, uo)
    is det.
:- mode foldl2(pred(in, in, out, in, out) is semidet, in,
    in, out, in, out) is semidet.
:- mode foldl2(pred(in, in, out, mdi, muo) is semidet, in,
    in, out, mdi, muo) is semidet.
:- mode foldl2(pred(in, in, out, di, uo) is semidet, in,
    in, out, di, uo) is semidet.

:- pred foldl3(pred(T1, T2, T2, T3, T3, T4, T4), array(T1),
    T2, T2, T3, T3, T4, T4).
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

:- pred foldl4(pred(T1, T2, T2, T3, T3, T4, T4, T5, T5), array(T1),
    T2, T2, T3, T3, T4, T4, T5, T5).
:- mode foldl4(pred(in, in, out, in, out, in, out, in, out) is det,
    in, in, out, in, out, in, out, in, out) is det.
:- mode foldl4(pred(in, in, out, in, out, in, out, mdi, muo) is det,
    in, in, out, in, out, in, out, mdi, muo) is det.
:- mode foldl4(pred(in, in, out, in, out, in, out, di, uo) is det,
    in, in, out, in, out, in, out, di, uo) is det.
:- mode foldl4(pred(in, in, out, in, out, in, out, in, out) is semidet,
    in, in, out, in, out, in, out, in, out) is semidet.
:- mode foldl4(pred(in, in, out, in, out, in, out, mdi, muo) is semidet,
    in, in, out, in, out, in, out, mdi, muo) is semidet.
:- mode foldl4(pred(in, in, out, in, out, in, out, di, uo) is semidet,
    in, in, out, in, out, in, out, di, uo) is semidet.

:- pred foldl5(pred(T1, T2, T2, T3, T3, T4, T4, T5, T5, T6, T6),
    array(T1), T2, T2, T3, T3, T4, T4, T5, T5, T6, T6).
:- mode foldl5(
    pred(in, in, out, in, out, in, out, in, out, in, out) is det,
    in, in, out, in, out, in, out, in, out, in, out) is det.
:- mode foldl5(
    pred(in, in, out, in, out, in, out, in, out, mdi, muo) is det,
    in, in, out, in, out, in, out, in, out, mdi, muo) is det.
:- mode foldl5(
    pred(in, in, out, in, out, in, out, in, out, di, uo) is det,
    in, in, out, in, out, in, out, in, out, di, uo) is det.
:- mode foldl5(
    pred(in, in, out, in, out, in, out, in, out, in, out) is semidet,
    in, in, out, in, out, in, out, in, out, in, out) is semidet.
:- mode foldl5(
    pred(in, in, out, in, out, in, out, in, out, mdi, muo) is semidet,
    in, in, out, in, out, in, out, in, out, mdi, muo) is semidet.
:- mode foldl5(
    pred(in, in, out, in, out, in, out, in, out, di, uo) is semidet,
    in, in, out, in, out, in, out, in, out, di, uo) is semidet.

:- func foldr(func(T1, T2) = T2, array(T1), T2) = T2.
:- mode foldr(func(in, in) = out is det, in, in) = out is det.
:- mode foldr(func(in, di) = uo is det, in, di) = uo is det.

:- pred foldr(pred(T1, T2, T2), array(T1), T2, T2).
:- mode foldr(pred(in, in, out) is det, in, in, out) is det.
:- mode foldr(pred(in, mdi, muo) is det, in, mdi, muo) is det.
:- mode foldr(pred(in, di, uo) is det, in, di, uo) is det.
:- mode foldr(pred(in, in, out) is semidet, in, in, out) is semidet.
:- mode foldr(pred(in, mdi, muo) is semidet, in, mdi, muo) is semidet.
:- mode foldr(pred(in, di, uo) is semidet, in, di, uo) is semidet.

:- pred foldr2(pred(T1, T2, T2, T3, T3), array(T1), T2, T2, T3, T3).
:- mode foldr2(pred(in, in, out, in, out) is det, in, in, out, in, out)
    is det.
:- mode foldr2(pred(in, in, out, mdi, muo) is det, in, in, out, mdi, muo)
    is det.
:- mode foldr2(pred(in, in, out, di, uo) is det, in, in, out, di, uo)
    is det.
:- mode foldr2(pred(in, in, out, in, out) is semidet, in,
    in, out, in, out) is semidet.
:- mode foldr2(pred(in, in, out, mdi, muo) is semidet, in,
    in, out, mdi, muo) is semidet.
:- mode foldr2(pred(in, in, out, di, uo) is semidet, in,
    in, out, di, uo) is semidet.

:- pred foldr3(pred(T1, T2, T2, T3, T3, T4, T4), array(T1),
    T2, T2, T3, T3, T4, T4).
:- mode foldr3(pred(in, in, out, in, out, in, out) is det, in,
    in, out, in, out, in, out) is det.
:- mode foldr3(pred(in, in, out, in, out, mdi, muo) is det, in,
    in, out, in, out, mdi, muo) is det.
:- mode foldr3(pred(in, in, out, in, out, di, uo) is det, in,
    in, out, in, out, di, uo) is det.
:- mode foldr3(pred(in, in, out, in, out, in, out) is semidet, in,
    in, out, in, out, in, out) is semidet.
:- mode foldr3(pred(in, in, out, in, out, mdi, muo) is semidet, in,
    in, out, in, out, mdi, muo) is semidet.
:- mode foldr3(pred(in, in, out, in, out, di, uo) is semidet, in,
    in, out, in, out, di, uo) is semidet.

:- pred foldr4(pred(T1, T2, T2, T3, T3, T4, T4, T5, T5), array(T1),
    T2, T2, T3, T3, T4, T4, T5, T5).
:- mode foldr4(pred(in, in, out, in, out, in, out, in, out) is det,
    in, in, out, in, out, in, out, in, out) is det.
:- mode foldr4(pred(in, in, out, in, out, in, out, mdi, muo) is det,
    in, in, out, in, out, in, out, mdi, muo) is det.
:- mode foldr4(pred(in, in, out, in, out, in, out, di, uo) is det,
    in, in, out, in, out, in, out, di, uo) is det.
:- mode foldr4(pred(in, in, out, in, out, in, out, in, out) is semidet,
    in, in, out, in, out, in, out, in, out) is semidet.
:- mode foldr4(pred(in, in, out, in, out, in, out, mdi, muo) is semidet,
    in, in, out, in, out, in, out, mdi, muo) is semidet.
:- mode foldr4(pred(in, in, out, in, out, in, out, di, uo) is semidet,
    in, in, out, in, out, in, out, di, uo) is semidet.

:- pred foldr5(pred(T1, T2, T2, T3, T3, T4, T4, T5, T5, T6, T6),
    array(T1), T2, T2, T3, T3, T4, T4, T5, T5, T6, T6).
:- mode foldr5(
    pred(in, in, out, in, out, in, out, in, out, in, out) is det,
    in, in, out, in, out, in, out, in, out, in, out) is det.
:- mode foldr5(
    pred(in, in, out, in, out, in, out, in, out, mdi, muo) is det,
    in, in, out, in, out, in, out, in, out, mdi, muo) is det.
:- mode foldr5(
    pred(in, in, out, in, out, in, out, in, out, di, uo) is det,
    in, in, out, in, out, in, out, in, out, di, uo) is det.
:- mode foldr5(
    pred(in, in, out, in, out, in, out, in, out, in, out) is semidet,
    in, in, out, in, out, in, out, in, out, in, out) is semidet.
:- mode foldr5(
    pred(in, in, out, in, out, in, out, in, out, mdi, muo) is semidet,
    in, in, out, in, out, in, out, in, out, mdi, muo) is semidet.
:- mode foldr5(
    pred(in, in, out, in, out, in, out, in, out, di, uo) is semidet,
    in, in, out, in, out, in, out, in, out, di, uo) is semidet.

:- pred foldl_corresponding(pred(T1, T2, T3, T3), array(T1), array(T2),
    T3, T3).
:- mode foldl_corresponding(in(pred(in, in, in, out) is det), in, in,
    in, out) is det.
:- mode foldl_corresponding(in(pred(in, in, mdi, muo) is det), in, in,
    mdi, muo) is det.
:- mode foldl_corresponding(in(pred(in, in, di, uo) is det), in, in,
    di, uo) is det.
:- mode foldl_corresponding(in(pred(in, in, in, out) is semidet), in, in,
    in, out) is semidet.
:- mode foldl_corresponding(in(pred(in, in, mdi, muo) is semidet), in, in,
    mdi, muo) is semidet.
:- mode foldl_corresponding(in(pred(in, in, di, uo) is semidet), in, in,
    di, uo) is semidet.

:- pred foldl2_corresponding(pred(T1, T2, T3, T3, T4, T4),
    array(T1), array(T2), T3, T3, T4, T4).
:- mode foldl2_corresponding(in(pred(in, in, in, out, in, out) is det),
    in, in, in, out, in, out) is det.
:- mode foldl2_corresponding(in(pred(in, in, in, out, mdi, muo) is det),
    in, in, in, out, mdi, muo) is det.
:- mode foldl2_corresponding(in(pred(in, in, in, out, di, uo) is det),
    in, in, in, out, di, uo) is det.
:- mode foldl2_corresponding(in(pred(in, in, in, out, in, out) is semidet),
    in, in, in, out, in, out) is semidet.
:- mode foldl2_corresponding(in(pred(in, in, in, out, mdi, muo) is semidet),
    in, in, in, out, mdi, muo) is semidet.
:- mode foldl2_corresponding(in(pred(in, in, in, out,  di, uo) is semidet),
    in, in, in, out, di, uo) is semidet.

:- pred map_foldl(pred(T1, T2, T3, T3), array(T1), array(T2), T3, T3).
:- mode map_foldl(in(pred(in, out, in, out) is det),
    in, array_uo, in, out) is det.
:- mode map_foldl(in(pred(in, out, mdi, muo) is det),
    in, array_uo, mdi, muo) is det.
:- mode map_foldl(in(pred(in, out, di, uo) is det),
    in, array_uo, di, uo) is det.
:- mode map_foldl(in(pred(in, out, in, out) is semidet),
    in, array_uo, in, out) is semidet.

:- pred map_corresponding_foldl(pred(T1, T2, T3, T4, T4),
    array(T1), array(T2), array(T3), T4, T4).
:- mode map_corresponding_foldl(
    in(pred(in, in, out, in, out) is det),
    in, in, array_uo, in, out) is det.
:- mode map_corresponding_foldl(
    in(pred(in, in, out, mdi, muo) is det),
    in, in, array_uo, mdi, muo) is det.
:- mode map_corresponding_foldl(
    in(pred(in, in, out, di, uo) is det),
    in, in, array_uo, di, uo) is det.
:- mode map_corresponding_foldl(
    in(pred(in, in, out, in, out) is semidet),
    in, in, array_uo, in, out) is semidet.
:- mode map_corresponding_foldl(
    in(pred(in, in, out, mdi, muo) is semidet),
    in, in, array_uo, mdi, muo) is semidet.
:- mode map_corresponding_foldl(
    in(pred(in, in, out, di, uo) is semidet),
    in, in, array_uo, di, uo) is semidet.

:- pred all_true(pred(T), array(T)).
:- mode all_true(in(pred(in) is semidet), in) is semidet.

:- pred all_false(pred(T), array(T)).
:- mode all_false(in(pred(in) is semidet), in) is semidet.

:- func append(array(T)::in, array(T)::in) = (array(T)::array_uo) is det.

%---------------------------------------------------------------------------%

:- implementation.
:- import_module int.
:- import_module require.

% Temporary (chunk 1 step 6, docs/INPROGRESS.md): only generate, its folds,
% map, to_list and the foldl family are compiled; the predicates code generation builds
% in are called through it, and the others are reported as not yet
% implemented where they are called. Step 6 (d) compiles the rest.

generate(Size, Func) = Array :-
    ( if Size < 0 then
        unexpected("function `array.generate'/2", "negative size")
    else if Size = 0 then
        make_empty_array(Array)
    else
        init(Size, Func(0), Array0),
        generate_from(1, Size, Func, Array0, Array)
    ).

to_list(Array, List) :-
    to_list_before(Array, size(Array), [], List).

to_list(Array) = List :-
    to_list(Array, List).

    % The elements before element I, in order, then List0.
    %
:- pred to_list_before(array(T)::in, int::in, list(T)::in, list(T)::out)
    is det.

to_list_before(Array, I, List0, List) :-
    ( if I =< 0 then
        List = List0
    else
        to_list_before(Array, I - 1, [Array ^ unsafe_elem(I - 1) | List0],
            List)
    ).

map(Pred, Array0, Array) :-
    Size = size(Array0),
    ( if Size = 0 then
        make_empty_array(Array)
    else
        Pred(Array0 ^ unsafe_elem(0), First),
        init(Size, First, Array1),
        map_from(1, Size, Pred, Array0, Array1, Array)
    ).

map(Func, Array0) = Array :-
    map(apply_func(Func), Array0, Array).

:- pred apply_func((func(T1) = T2)::in, T1::in, T2::out) is det.

apply_func(Func, X, Func(X)).

    % Element I of the new array is Pred's of element I of Array0, from I on.
    %
:- pred map_from(int::in, int::in, pred(T1, T2)::in(pred(in, out) is det),
    array(T1)::in, array(T2)::array_di, array(T2)::array_uo) is det.

map_from(I, Size, Pred, Array0, !Array) :-
    ( if I >= Size then
        true
    else
        Pred(Array0 ^ unsafe_elem(I), X),
        unsafe_set(I, X, !Array),
        map_from(I + 1, Size, Pred, Array0, !Array)
    ).

:- pred generate_from(int::in, int::in, (func(int) = T)::in,
    array(T)::array_di, array(T)::array_uo) is det.

generate_from(I, Size, Func, !Array) :-
    ( if I >= Size then
        true
    else
        unsafe_set(I, Func(I), !Array),
        generate_from(I + 1, Size, Func, !Array)
    ).

:- pred generate_foldl_from(int, int, pred(int, T, A, A),
    array(T), array(T), A, A).
:- mode generate_foldl_from(in, in, in(pred(in, out, in, out) is det),
    array_di, array_uo, in, out) is det.
:- mode generate_foldl_from(in, in, in(pred(in, out, mdi, muo) is det),
    array_di, array_uo, mdi, muo) is det.
:- mode generate_foldl_from(in, in, in(pred(in, out, di, uo) is det),
    array_di, array_uo, di, uo) is det.
:- mode generate_foldl_from(in, in, in(pred(in, out, in, out) is semidet),
    array_di, array_uo, in, out) is semidet.
:- mode generate_foldl_from(in, in, in(pred(in, out, mdi, muo) is semidet),
    array_di, array_uo, mdi, muo) is semidet.
:- mode generate_foldl_from(in, in, in(pred(in, out, di, uo) is semidet),
    array_di, array_uo, di, uo) is semidet.

generate_foldl(Size, Pred, Array, !Acc1) :-
    ( if Size < 0 then
        unexpected("predicate `array.generate_foldl'/5", "negative size")
    else if Size = 0 then
        make_empty_array(Array)
    else
        Pred(0, X, !Acc1),
        init(Size, X, Array0),
        generate_foldl_from(1, Size, Pred, Array0, Array, !Acc1)
    ).

generate_foldl_from(I, Size, Pred, !Array, !Acc1) :-
    ( if I >= Size then
        true
    else
        Pred(I, X, !Acc1),
        unsafe_set(I, X, !Array),
        generate_foldl_from(I + 1, Size, Pred, !Array, !Acc1)
    ).

:- pred generate_foldl2_from(int, int, pred(int, T, A, A, B, B),
    array(T), array(T), A, A, B, B).
:- mode generate_foldl2_from(in, in, in(pred(in, out, in, out, in, out) is det),
    array_di, array_uo, in, out, in, out) is det.
:- mode generate_foldl2_from(in, in, in(pred(in, out, mdi, muo, mdi, muo) is det),
    array_di, array_uo, mdi, muo, mdi, muo) is det.
:- mode generate_foldl2_from(in, in, in(pred(in, out, di, uo, di, uo) is det),
    array_di, array_uo, di, uo, di, uo) is det.
:- mode generate_foldl2_from(in, in, in(pred(in, out, in, out, in, out) is semidet),
    array_di, array_uo, in, out, in, out) is semidet.
:- mode generate_foldl2_from(in, in, in(pred(in, out, mdi, muo, mdi, muo) is semidet),
    array_di, array_uo, mdi, muo, mdi, muo) is semidet.
:- mode generate_foldl2_from(in, in, in(pred(in, out, di, uo, di, uo) is semidet),
    array_di, array_uo, di, uo, di, uo) is semidet.

generate_foldl2(Size, Pred, Array, !Acc1, !Acc2) :-
    ( if Size < 0 then
        unexpected("predicate `array.generate_foldl2'/7", "negative size")
    else if Size = 0 then
        make_empty_array(Array)
    else
        Pred(0, X, !Acc1, !Acc2),
        init(Size, X, Array0),
        generate_foldl2_from(1, Size, Pred, Array0, Array, !Acc1, !Acc2)
    ).

generate_foldl2_from(I, Size, Pred, !Array, !Acc1, !Acc2) :-
    ( if I >= Size then
        true
    else
        Pred(I, X, !Acc1, !Acc2),
        unsafe_set(I, X, !Array),
        generate_foldl2_from(I + 1, Size, Pred, !Array, !Acc1, !Acc2)
    ).

foldl(Func, Array, Acc0) = Acc :-
    foldl_func_from(Func, Array, 0, size(Array), Acc0, Acc).

:- pred foldl_func_from(func(T1, T2) = T2, array(T1), int, int, T2, T2).
:- mode foldl_func_from(func(in, in) = out is det, in, in, in, in, out)
    is det.
:- mode foldl_func_from(func(in, di) = uo is det, in, in, in, di, uo)
    is det.

foldl_func_from(Func, Array, I, Size, !Acc) :-
    ( if I >= Size then
        true
    else
        unsafe_lookup(Array, I, X),
        !:Acc = Func(X, !.Acc),
        foldl_func_from(Func, Array, I + 1, Size, !Acc)
    ).

:- pred foldl_from(pred(T1, T2, T2), array(T1), int, int,
    T2, T2).
:- mode foldl_from(pred(in, in, out) is det, in, in, in,
    in, out) is det.
:- mode foldl_from(pred(in, mdi, muo) is det, in, in, in,
    mdi, muo) is det.
:- mode foldl_from(pred(in, di, uo) is det, in, in, in,
    di, uo) is det.
:- mode foldl_from(pred(in, in, out) is semidet, in, in, in,
    in, out) is semidet.
:- mode foldl_from(pred(in, mdi, muo) is semidet, in, in, in,
    mdi, muo) is semidet.
:- mode foldl_from(pred(in, di, uo) is semidet, in, in, in,
    di, uo) is semidet.

foldl_from(Pred, Array, I, Size, !Acc1) :-
    ( if I >= Size then
        true
    else
        unsafe_lookup(Array, I, X),
        Pred(X, !Acc1),
        foldl_from(Pred, Array, I + 1, Size, !Acc1)
    ).

foldl(Pred, Array, !Acc1) :-
    foldl_from(Pred, Array, 0, size(Array), !Acc1).

:- pred foldl2_from(pred(T1, T2, T2, T3, T3), array(T1), int, int,
    T2, T2, T3, T3).
:- mode foldl2_from(pred(in, in, out, in, out) is det, in, in, in,
    in, out, in, out) is det.
:- mode foldl2_from(pred(in, in, out, mdi, muo) is det, in, in, in,
    in, out, mdi, muo) is det.
:- mode foldl2_from(pred(in, in, out, di, uo) is det, in, in, in,
    in, out, di, uo) is det.
:- mode foldl2_from(pred(in, in, out, in, out) is semidet, in, in, in,
    in, out, in, out) is semidet.
:- mode foldl2_from(pred(in, in, out, mdi, muo) is semidet, in, in, in,
    in, out, mdi, muo) is semidet.
:- mode foldl2_from(pred(in, in, out, di, uo) is semidet, in, in, in,
    in, out, di, uo) is semidet.

foldl2_from(Pred, Array, I, Size, !Acc1, !Acc2) :-
    ( if I >= Size then
        true
    else
        unsafe_lookup(Array, I, X),
        Pred(X, !Acc1, !Acc2),
        foldl2_from(Pred, Array, I + 1, Size, !Acc1, !Acc2)
    ).

foldl2(Pred, Array, !Acc1, !Acc2) :-
    foldl2_from(Pred, Array, 0, size(Array), !Acc1, !Acc2).

:- pred foldl3_from(pred(T1, T2, T2, T3, T3, T4, T4), array(T1), int, int,
    T2, T2, T3, T3, T4, T4).
:- mode foldl3_from(pred(in, in, out, in, out, in, out) is det, in, in, in,
    in, out, in, out, in, out) is det.
:- mode foldl3_from(pred(in, in, out, in, out, mdi, muo) is det, in, in, in,
    in, out, in, out, mdi, muo) is det.
:- mode foldl3_from(pred(in, in, out, in, out, di, uo) is det, in, in, in,
    in, out, in, out, di, uo) is det.
:- mode foldl3_from(pred(in, in, out, in, out, in, out) is semidet, in, in, in,
    in, out, in, out, in, out) is semidet.
:- mode foldl3_from(pred(in, in, out, in, out, mdi, muo) is semidet, in, in, in,
    in, out, in, out, mdi, muo) is semidet.
:- mode foldl3_from(pred(in, in, out, in, out, di, uo) is semidet, in, in, in,
    in, out, in, out, di, uo) is semidet.

foldl3_from(Pred, Array, I, Size, !Acc1, !Acc2, !Acc3) :-
    ( if I >= Size then
        true
    else
        unsafe_lookup(Array, I, X),
        Pred(X, !Acc1, !Acc2, !Acc3),
        foldl3_from(Pred, Array, I + 1, Size, !Acc1, !Acc2, !Acc3)
    ).

foldl3(Pred, Array, !Acc1, !Acc2, !Acc3) :-
    foldl3_from(Pred, Array, 0, size(Array), !Acc1, !Acc2, !Acc3).

:- pred foldl4_from(pred(T1, T2, T2, T3, T3, T4, T4, T5, T5), array(T1), int, int,
    T2, T2, T3, T3, T4, T4, T5, T5).
:- mode foldl4_from(pred(in, in, out, in, out, in, out, in, out) is det, in, in, in,
    in, out, in, out, in, out, in, out) is det.
:- mode foldl4_from(pred(in, in, out, in, out, in, out, mdi, muo) is det, in, in, in,
    in, out, in, out, in, out, mdi, muo) is det.
:- mode foldl4_from(pred(in, in, out, in, out, in, out, di, uo) is det, in, in, in,
    in, out, in, out, in, out, di, uo) is det.
:- mode foldl4_from(pred(in, in, out, in, out, in, out, in, out) is semidet, in, in, in,
    in, out, in, out, in, out, in, out) is semidet.
:- mode foldl4_from(pred(in, in, out, in, out, in, out, mdi, muo) is semidet, in, in, in,
    in, out, in, out, in, out, mdi, muo) is semidet.
:- mode foldl4_from(pred(in, in, out, in, out, in, out, di, uo) is semidet, in, in, in,
    in, out, in, out, in, out, di, uo) is semidet.

foldl4_from(Pred, Array, I, Size, !Acc1, !Acc2, !Acc3, !Acc4) :-
    ( if I >= Size then
        true
    else
        unsafe_lookup(Array, I, X),
        Pred(X, !Acc1, !Acc2, !Acc3, !Acc4),
        foldl4_from(Pred, Array, I + 1, Size, !Acc1, !Acc2, !Acc3, !Acc4)
    ).

foldl4(Pred, Array, !Acc1, !Acc2, !Acc3, !Acc4) :-
    foldl4_from(Pred, Array, 0, size(Array), !Acc1, !Acc2, !Acc3, !Acc4).

:- pred foldl5_from(pred(T1, T2, T2, T3, T3, T4, T4, T5, T5, T6, T6), array(T1), int, int,
    T2, T2, T3, T3, T4, T4, T5, T5, T6, T6).
:- mode foldl5_from(pred(in, in, out, in, out, in, out, in, out, in, out) is det, in, in, in,
    in, out, in, out, in, out, in, out, in, out) is det.
:- mode foldl5_from(pred(in, in, out, in, out, in, out, in, out, mdi, muo) is det, in, in, in,
    in, out, in, out, in, out, in, out, mdi, muo) is det.
:- mode foldl5_from(pred(in, in, out, in, out, in, out, in, out, di, uo) is det, in, in, in,
    in, out, in, out, in, out, in, out, di, uo) is det.
:- mode foldl5_from(pred(in, in, out, in, out, in, out, in, out, in, out) is semidet, in, in, in,
    in, out, in, out, in, out, in, out, in, out) is semidet.
:- mode foldl5_from(pred(in, in, out, in, out, in, out, in, out, mdi, muo) is semidet, in, in, in,
    in, out, in, out, in, out, in, out, mdi, muo) is semidet.
:- mode foldl5_from(pred(in, in, out, in, out, in, out, in, out, di, uo) is semidet, in, in, in,
    in, out, in, out, in, out, in, out, di, uo) is semidet.

foldl5_from(Pred, Array, I, Size, !Acc1, !Acc2, !Acc3, !Acc4, !Acc5) :-
    ( if I >= Size then
        true
    else
        unsafe_lookup(Array, I, X),
        Pred(X, !Acc1, !Acc2, !Acc3, !Acc4, !Acc5),
        foldl5_from(Pred, Array, I + 1, Size, !Acc1, !Acc2, !Acc3, !Acc4, !Acc5)
    ).

foldl5(Pred, Array, !Acc1, !Acc2, !Acc3, !Acc4, !Acc5) :-
    foldl5_from(Pred, Array, 0, size(Array), !Acc1, !Acc2, !Acc3, !Acc4, !Acc5).
