% A declaration-only stub of the standard library module builtin, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module builtin.
:- interface.

:- type c_pointer.
:- inst dead == clobbered.
:- inst mostly_dead == mostly_clobbered.
:- mode unused == free >> free.
:- mode output == free >> ground.
:- mode input  == ground >> ground.
:- mode in  == ground >> ground.
:- mode out == free >> ground.
:- mode in(Inst)  == Inst >> Inst.
:- mode out(Inst) == free >> Inst.
:- mode di(Inst)  == Inst >> clobbered.
:- mode mdi(Inst) == Inst >> mostly_clobbered.
:- mode uo == free >> unique.
:- mode ui == unique >> unique.
:- mode di == unique >> clobbered.
:- mode muo == free >> mostly_unique.
:- mode mui == mostly_unique >> mostly_unique.
:- mode mdi == mostly_unique >> mostly_clobbered.
:- mode ia == any >> any.
:- mode oa == free >> any.
:- pred copy(T, T).
:- mode copy(ui, uo) is det.
:- mode copy(in, uo) is det.
:- func unsafe_promise_unique(T::in) = (T::uo) is det.
:- pred unsafe_promise_unique(T::in, T::uo) is det.
:- pred false is failure.
:- func unsafe_cast_any_to_ground(T::ia) = (T::out) is det.
:- pred unify(T::in, T::in) is semidet.
:- type unify(T) == pred(T, T).
:- inst unify == (pred(in, in) is semidet).
:- type comparison_result
    --->    (=)
    ;       (<)
    ;       (>).
:- pred compare(comparison_result, T, T).
:- mode compare(uo, in, in) is det.
:- mode compare(uo, ui, ui) is det.
:- mode compare(uo, ui, in) is det.
:- mode compare(uo, in, ui) is det.
:- type compare(T) == pred(comparison_result, T, T).
:- inst compare == (pred(uo, in, in) is det).
:- func ordering(T, T) = comparison_result.
:- pred T  @<  T.
:- mode in @< in is semidet.
:- pred T  @=<  T.
:- mode in @=< in is semidet.
:- pred T  @>  T.
:- mode in @> in is semidet.
:- pred T  @>=  T.
:- mode in @>= in is semidet.
:- type comparison_pred(T) == pred(T, T, comparison_result).
:- inst comparison_pred(I) == (pred(in(I), in(I), out) is det).
:- inst comparison_pred == comparison_pred(ground).
:- type comparison_func(T) == (func(T, T) = comparison_result).
:- inst comparison_func(I) == (func(in(I), in(I)) = out is det).
:- inst comparison_func == comparison_func(ground).
:- pred semidet_succeed is semidet.
:- pred semidet_fail is semidet.
:- pred semidet_true is semidet.
:- pred semidet_false is semidet.
:- pred cc_multi_equal(T, T).
:- mode cc_multi_equal(di, uo) is cc_multi.
:- mode cc_multi_equal(in, out) is cc_multi.
:- impure pred impure_true is det.
:- semipure pred semipure_true is det.
:- pred dynamic_cast(T1::in, T2::out) is semidet.
