% A declaration-only stub of the standard library module int, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module int.
:- interface.

:- import_module array.

:- pred (int::in) < (int::in) is semidet.
:- pred (int::in) > (int::in) is semidet.
:- pred (int::in) =< (int::in) is semidet.
:- pred (int::in) >= (int::in) is semidet.
:- func abs(int) = int.
:- pred abs(int::in, int::out) is det.
:- func unchecked_abs(int) = int.
:- func nabs(int) = int.
:- func max(int, int) = int.
:- pred max(int::in, int::in, int::out) is det.
:- func min(int, int) = int.
:- pred min(int::in, int::in, int::out) is det.
:- func + (int::in) = (int::uo) is det.
:- func - (int::in) = (int::uo) is det.
:- func int + int = int.
:- mode in  + in  = uo  is det.
:- mode uo  + in  = in  is det.
:- mode in  + uo  = in  is det.
:- func plus(int, int) = int.
:- func int - int = int.
:- mode in  - in  = uo  is det.
:- mode uo  - in  = in  is det.
:- mode in  - uo  = in  is det.
:- func minus(int, int) = int.
:- func (int::in) * (int::in) = (int::uo) is det.
:- func times(int, int) = int.
:- func div(int::in, int::in) = (int::uo) is det.
:- func (int::in) // (int::in) = (int::uo) is det.
:- func (int::in) / (int::in) = (int::uo) is det.
:- func unchecked_quotient(int::in, int::in) = (int::uo) is det.
:- func (int::in) mod (int::in) = (int::uo) is det.
:- func (int::in) rem (int::in) = (int::uo) is det.
:- func unchecked_rem(int::in, int::in) = (int::uo) is det.
:- pred even(int::in) is semidet.
:- pred odd(int::in) is semidet.
:- func pow(int, int) = int.
:- pred pow(int::in, int::in, int::out) is det.
:- func log2(int) = int.
:- pred log2(int::in, int::out) is det.
:- func (int::in) << (int::in) = (int::uo) is det.
:- func unchecked_left_shift(int::in, int::in) = (int::uo) is det.
:- func (int::in) >> (int::in) = (int::uo) is det.
:- func unchecked_right_shift(int::in, int::in) = (int::uo) is det.
:- func \ (int::in) = (int::uo) is det.
:- func (int::in) /\ (int::in) = (int::uo) is det.
:- func (int::in) \/ (int::in) = (int::uo) is det.
:- func xor(int, int) = int.
:- mode xor(in, in) = uo is det.
:- mode xor(in, uo) = in is det.
:- mode xor(uo, in) = in is det.
:- func max_int = int.
:- pred max_int(int::out) is det.
:- func min_int = int.
:- pred min_int(int::out) is det.
:- func bits_per_int = int.
:- pred bits_per_int(int::out) is det.
:- func fold_up(func(int, T) = T, int, int, T) = T.
:- pred fold_up(pred(int, T, T), int, int, T, T).
:- mode fold_up(pred(in, in, out) is det, in, in, in, out) is det.
:- mode fold_up(pred(in, mdi, muo) is det, in, in, mdi, muo) is det.
:- mode fold_up(pred(in, di, uo) is det, in, in, di, uo) is det.
:- mode fold_up(pred(in, array_di, array_uo) is det, in, in,
    array_di, array_uo) is det.
:- mode fold_up(pred(in, in, out) is semidet, in, in, in, out)
    is semidet.
:- mode fold_up(pred(in, mdi, muo) is semidet, in, in, mdi, muo)
    is semidet.
:- mode fold_up(pred(in, di, uo) is semidet, in, in, di, uo)
    is semidet.
:- mode fold_up(pred(in, in, out) is nondet, in, in, in, out)
    is nondet.
:- mode fold_up(pred(in, mdi, muo) is nondet, in, in, mdi, muo)
    is nondet.
:- mode fold_up(pred(in, di, uo) is cc_multi, in, in, di, uo)
    is cc_multi.
:- mode fold_up(pred(in, in, out) is cc_multi, in, in, in, out)
    is cc_multi.
:- pred fold_up2(pred(int, T, T, U, U), int, int, T, T, U, U).
:- mode fold_up2(pred(in, in, out, in, out) is det, in, in, in, out,
    in, out) is det.
:- mode fold_up2(pred(in, in, out, mdi, muo) is det, in, in, in, out,
    mdi, muo) is det.
:- mode fold_up2(pred(in, in, out, di, uo) is det, in, in, in, out,
    di, uo) is det.
:- mode fold_up2(pred(in, di, uo, di, uo) is det, in, in, di, uo,
    di, uo) is det.
:- mode fold_up2(pred(in, in, out, array_di, array_uo) is det, in, in,
    in, out, array_di, array_uo) is det.
:- mode fold_up2(pred(in, in, out, in, out) is semidet, in, in,
    in, out, in, out) is semidet.
:- mode fold_up2(pred(in, in, out, mdi, muo) is semidet, in, in,
    in, out, mdi, muo) is semidet.
:- mode fold_up2(pred(in, in, out, di, uo) is semidet, in, in,
    in, out, di, uo) is semidet.
:- mode fold_up2(pred(in, in, out, in, out) is nondet, in, in,
    in, out, in, out) is nondet.
:- mode fold_up2(pred(in, in, out, mdi, muo) is nondet, in, in,
    in, out, mdi, muo) is nondet.
:- pred fold_up3(pred(int, T, T, U, U, V, V), int, int, T, T, U, U, V, V).
:- mode fold_up3(pred(in, in, out, in, out, in, out) is det,
    in, in, in, out, in, out, in, out) is det.
:- mode fold_up3(pred(in, in, out, in, out, mdi, muo) is det,
    in, in, in, out, in, out, mdi, muo) is det.
:- mode fold_up3(pred(in, in, out, in, out, di, uo) is det,
    in, in, in, out, in, out, di, uo) is det.
:- mode fold_up3(pred(in, in, out, di, uo, di, uo) is det,
    in, in, in, out, di, uo, di, uo) is det.
:- mode fold_up3(pred(in, in, out, in, out, array_di, array_uo) is det,
    in, in, in, out, in, out, array_di, array_uo) is det.
:- mode fold_up3(pred(in, in, out, in, out, in, out) is semidet,
    in, in, in, out, in, out, in, out) is semidet.
:- mode fold_up3(pred(in, in, out, in, out, mdi, muo) is semidet,
    in, in, in, out, in, out, mdi, muo) is semidet.
:- mode fold_up3(pred(in, in, out, in, out, di, uo) is semidet,
    in, in, in, out, in, out, di, uo) is semidet.
:- mode fold_up3(pred(in, in, out, in, out, in, out) is nondet,
    in, in, in, out, in, out, in, out) is nondet.
:- mode fold_up3(pred(in, in, out, in, out, mdi, muo) is nondet,
    in, in, in, out, in, out, mdi, muo) is nondet.
:- func fold_down(func(int, T) = T, int, int, T) = T.
:- pred fold_down(pred(int, T, T), int, int, T, T).
:- mode fold_down(pred(in, in, out) is det, in, in, in, out) is det.
:- mode fold_down(pred(in, mdi, muo) is det, in, in, mdi, muo) is det.
:- mode fold_down(pred(in, di, uo) is det, in, in, di, uo) is det.
:- mode fold_down(pred(in, array_di, array_uo) is det, in, in,
    array_di, array_uo) is det.
:- mode fold_down(pred(in, in, out) is semidet, in, in, in, out)
    is semidet.
:- mode fold_down(pred(in, mdi, muo) is semidet, in, in, mdi, muo)
    is semidet.
:- mode fold_down(pred(in, di, uo) is semidet, in, in, di, uo)
    is semidet.
:- mode fold_down(pred(in, in, out) is nondet, in, in, in, out)
    is nondet.
:- mode fold_down(pred(in, mdi, muo) is nondet, in, in, mdi, muo)
    is nondet.
:- mode fold_down(pred(in, in, out) is cc_multi, in, in, in, out)
    is cc_multi.
:- mode fold_down(pred(in, di, uo) is cc_multi, in, in, di, uo)
    is cc_multi.
:- pred fold_down2(pred(int, T, T, U, U), int, int, T, T, U, U).
:- mode fold_down2(pred(in, in, out, in, out) is det, in, in, in, out,
    in, out) is det.
:- mode fold_down2(pred(in, in, out, mdi, muo) is det, in, in, in, out,
    mdi, muo) is det.
:- mode fold_down2(pred(in, in, out, di, uo) is det, in, in, in, out,
    di, uo) is det.
:- mode fold_down2(pred(in, di, uo, di, uo) is det, in, in, di, uo,
    di, uo) is det.
:- mode fold_down2(pred(in, in, out, array_di, array_uo) is det, in, in,
    in, out, array_di, array_uo) is det.
:- mode fold_down2(pred(in, in, out, in, out) is semidet, in, in,
    in, out, in, out) is semidet.
:- mode fold_down2(pred(in, in, out, di, uo) is semidet, in, in,
    in, out, di, uo) is semidet.
:- mode fold_down2(pred(in, in, out, in, out) is nondet, in, in,
    in, out, in, out) is nondet.
:- mode fold_down2(pred(in, in, out, mdi, muo) is nondet, in, in,
    in, out, mdi, muo) is nondet.
:- pred fold_down3(pred(int, T, T, U, U, V, V), int, int, T, T, U, U, V, V).
:- mode fold_down3(pred(in, in, out, in, out, in, out) is det,
    in, in, in, out, in, out, in, out) is det.
:- mode fold_down3(pred(in, in, out, in, out, mdi, muo) is det,
    in, in, in, out, in, out, mdi, muo) is det.
:- mode fold_down3(pred(in, in, out, in, out, di, uo) is det,
    in, in, in, out, in, out, di, uo) is det.
:- mode fold_down3(pred(in, in, out, di, uo, di, uo) is det,
    in, in, in, out, di, uo, di, uo) is det.
:- mode fold_down3(pred(in, in, out, in, out, array_di, array_uo) is det,
    in, in, in, out, in, out, array_di, array_uo) is det.
:- mode fold_down3(pred(in, in, out, in, out, in, out) is semidet,
    in, in, in, out, in, out, in, out) is semidet.
:- mode fold_down3(pred(in, in, out, in, out, mdi, muo) is semidet,
    in, in, in, out, in, out, mdi, muo) is semidet.
:- mode fold_down3(pred(in, in, out, in, out, di, uo) is semidet,
    in, in, in, out, in, out, di, uo) is semidet.
:- mode fold_down3(pred(in, in, out, in, out, in, out) is nondet,
    in, in, in, out, in, out, in, out) is nondet.
:- mode fold_down3(pred(in, in, out, in, out, mdi, muo) is nondet,
    in, in, in, out, in, out, mdi, muo) is nondet.
:- pred nondet_int_in_range(int::in, int::in, int::out) is nondet.
:- pred all_true_in_range(pred(int)::in(pred(in) is semidet),
    int::in, int::in) is semidet.
:- func hash(int) = int.
:- pred hash(int::in, int::out) is det.
