% The standard library module maybe, with the declarations of the library
% reference manual (22.01), all implemented in Mercury.

:- module maybe.
:- interface.
:- import_module list.

:- type maybe(T)
    --->    no
    ;       yes(T).
:- inst maybe(I) for maybe/1
    --->    no
    ;       yes(I).
:- inst maybe_yes(I) for maybe/1
    --->    yes(I).
:- type maybe_error
    --->    ok
    ;       error(string).
:- type maybe_error(T) == maybe_error(T, string).
:- type maybe_error(T, E)
    --->    ok(T)
    ;       error(E).
:- inst maybe_error(I) for maybe_error/2
    --->    ok(I)
    ;       error(ground).
:- inst maybe_error_ok(I) for maybe_error/2
    --->    ok(I).
:- type maybe_errors(T) == maybe_errors(T, string).
:- type maybe_errors(T, E)
    --->    ok(T)
    ;       error(E, list(E)).
:- inst maybe_errors_ok(I) for maybe_errors/2
    --->    ok(I).
:- func map_maybe(func(T) = U, maybe(T)) = maybe(U).
:- pred map_maybe(pred(T, U), maybe(T), maybe(U)).
:- mode map_maybe(pred(in, out) is det, in, out) is det.
:- mode map_maybe(pred(in, out) is semidet, in, out) is semidet.
:- mode map_maybe(pred(in, out) is multi, in, out) is multi.
:- mode map_maybe(pred(in, out) is nondet, in, out) is nondet.
:- func fold_maybe(func(T, U) = U, maybe(T), U) = U.
:- pred fold_maybe(pred(T, A, A), maybe(T), A, A).
:- mode fold_maybe(pred(in, in, out) is det, in, in, out) is det.
:- mode fold_maybe(pred(in, mdi, muo) is det, in, mdi, muo) is det.
:- mode fold_maybe(pred(in, di, uo) is det, in, di, uo) is det.
:- mode fold_maybe(pred(in, in, out) is semidet, in, in, out) is semidet.
:- mode fold_maybe(pred(in, mdi, muo) is semidet, in, mdi, muo) is semidet.
:- mode fold_maybe(pred(in, di, uo) is semidet, in, di, uo) is semidet.
:- pred fold2_maybe(pred(T, A, A, B, B), maybe(T), A, A, B, B).
:- mode fold2_maybe(pred(in, in, out, in, out) is det,
    in, in, out, in, out) is det.
:- mode fold2_maybe(pred(in, in, out, mdi, muo) is det,
    in, in, out, mdi, muo) is det.
:- mode fold2_maybe(pred(in, in, out, di, uo) is det,
    in, in, out, di, uo) is det.
:- mode fold2_maybe(pred(in, in, out, in, out) is semidet,
    in, in, out, in, out) is semidet.
:- mode fold2_maybe(pred(in, in, out, mdi, muo) is semidet,
    in, in, out, mdi, muo) is semidet.
:- mode fold2_maybe(pred(in, in, out, di, uo) is semidet,
    in, in, out, di, uo) is semidet.
:- pred fold3_maybe(pred(T, A, A, B, B, C, C), maybe(T), A, A, B, B, C, C).
:- mode fold3_maybe(pred(in, in, out, in, out, in, out) is det,
    in, in, out, in, out, in, out) is det.
:- mode fold3_maybe(pred(in, in, out, in, out, mdi, muo) is det,
    in, in, out, in, out, mdi, muo) is det.
:- mode fold3_maybe(pred(in, in, out, in, out, di, uo) is det,
    in, in, out, in, out, di, uo) is det.
:- mode fold3_maybe(pred(in, in, out, in, out, in, out) is semidet,
    in, in, out, in, out, in, out) is semidet.
:- mode fold3_maybe(pred(in, in, out, in, out, mdi, muo) is semidet,
    in, in, out, in, out, mdi, muo) is semidet.
:- mode fold3_maybe(pred(in, in, out, in, out, di, uo) is semidet,
    in, in, out, in, out, di, uo) is semidet.
:- pred fold4_maybe(pred(T, A, A, B, B, C, C, D, D),
    maybe(T), A, A, B, B, C, C, D, D).
:- mode fold4_maybe(pred(in, in, out, in, out, in, out, in, out) is det,
    in, in, out, in, out, in, out, in, out) is det.
:- mode fold4_maybe(pred(in, in, out, in, out, in, out, mdi, muo) is det,
    in, in, out, in, out, in, out, mdi, muo) is det.
:- mode fold4_maybe(pred(in, in, out, in, out, in, out, di, uo) is det,
    in, in, out, in, out, in, out, di, uo) is det.
:- mode fold4_maybe(pred(in, in, out, in, out, in, out, in, out) is semidet,
    in, in, out, in, out, in, out, in, out) is semidet.
:- mode fold4_maybe(pred(in, in, out, in, out, in, out, mdi, muo) is semidet,
    in, in, out, in, out, in, out, mdi, muo) is semidet.
:- mode fold4_maybe(pred(in, in, out, in, out, in, out, di, uo) is semidet,
    in, in, out, in, out, in, out, di, uo) is semidet.
:- pred fold5_maybe(pred(T, A, A, B, B, C, C, D, D, E, E),
    maybe(T), A, A, B, B, C, C, D, D, E, E).
:- mode fold5_maybe(
    pred(in, in, out, in, out, in, out, in, out, in, out) is det,
    in, in, out, in, out, in, out, in, out, in, out) is det.
:- mode fold5_maybe(
    pred(in, in, out, in, out, in, out, in, out, mdi, muo) is det,
    in, in, out, in, out, in, out, in, out, mdi, muo) is det.
:- mode fold5_maybe(
    pred(in, in, out, in, out, in, out, in, out, di, uo) is det,
    in, in, out, in, out, in, out, in, out, di, uo) is det.
:- mode fold5_maybe(
    pred(in, in, out, in, out, in, out, in, out, in, out) is semidet,
    in, in, out, in, out, in, out, in, out, in, out) is semidet.
:- mode fold5_maybe(
    pred(in, in, out, in, out, in, out, in, out, mdi, muo) is semidet,
    in, in, out, in, out, in, out, in, out, mdi, muo) is semidet.
:- mode fold5_maybe(
    pred(in, in, out, in, out, in, out, in, out, di, uo) is semidet,
    in, in, out, in, out, in, out, in, out, di, uo) is semidet.
:- pred map_fold_maybe(pred(T, U, A, A), maybe(T), maybe(U), A, A).
:- mode map_fold_maybe(pred(in, out, in, out) is det,
    in, out, in, out) is det.
:- mode map_fold_maybe(pred(in, out, mdi, muo) is det,
    in, out, mdi, muo) is det.
:- mode map_fold_maybe(pred(in, out, di, uo) is det,
    in, out, di, uo) is det.
:- mode map_fold_maybe(pred(in, out, in, out) is semidet,
    in, out, in, out) is semidet.
:- mode map_fold_maybe(pred(in, out, mdi, muo) is semidet,
    in, out, mdi, muo) is semidet.
:- mode map_fold_maybe(pred(in, out, di, uo) is semidet,
    in, out, di, uo) is semidet.
:- pred map_fold2_maybe(pred(T, U, A, A, B, B),
    maybe(T), maybe(U), A, A, B, B).
:- mode map_fold2_maybe(pred(in, out, in, out, in, out) is det,
    in, out, in, out, in, out) is det.
:- mode map_fold2_maybe(pred(in, out, in, out, mdi, muo) is det,
    in, out, in, out, mdi, muo) is det.
:- mode map_fold2_maybe(pred(in, out, in, out, di, uo) is det,
    in, out, in, out, di, uo) is det.
:- mode map_fold2_maybe(pred(in, out, in, out, in, out) is semidet,
    in, out, in, out, in, out) is semidet.
:- mode map_fold2_maybe(pred(in, out, in, out, mdi, muo) is semidet,
    in, out, in, out, mdi, muo) is semidet.
:- mode map_fold2_maybe(pred(in, out, in, out, di, uo) is semidet,
    in, out, in, out, di, uo) is semidet.
:- pred map_fold3_maybe(pred(T, U, A, A, B, B, C, C),
    maybe(T), maybe(U), A, A, B, B, C, C).
:- mode map_fold3_maybe(pred(in, out, in, out, in, out, in, out) is det,
    in, out, in, out, in, out, in, out) is det.
:- mode map_fold3_maybe(pred(in, out, in, out, in, out, mdi, muo) is det,
    in, out, in, out, in, out, mdi, muo) is det.
:- mode map_fold3_maybe(pred(in, out, in, out, in, out, di, uo) is det,
    in, out, in, out, in, out, di, uo) is det.
:- mode map_fold3_maybe(pred(in, out, in, out, in, out, in, out) is semidet,
    in, out, in, out, in, out, in, out) is semidet.
:- mode map_fold3_maybe(pred(in, out, in, out, in, out, mdi, muo) is semidet,
    in, out, in, out, in, out, mdi, muo) is semidet.
:- mode map_fold3_maybe(pred(in, out, in, out, in, out, di, uo) is semidet,
    in, out, in, out, in, out, di, uo) is semidet.
:- pred map_fold4_maybe(pred(T, U, A, A, B, B, C, C, D, D),
    maybe(T), maybe(U), A, A, B, B, C, C, D, D).
:- mode map_fold4_maybe(
    pred(in, out, in, out, in, out, in, out, in, out) is det,
    in, out, in, out, in, out, in, out, in, out) is det.
:- mode map_fold4_maybe(
    pred(in, out, in, out, in, out, in, out, mdi, muo) is det,
    in, out, in, out, in, out, in, out, mdi, muo) is det.
:- mode map_fold4_maybe(
    pred(in, out, in, out, in, out, in, out, di, uo) is det,
    in, out, in, out, in, out, in, out, di, uo) is det.
:- mode map_fold4_maybe(
    pred(in, out, in, out, in, out, in, out, in, out) is semidet,
    in, out, in, out, in, out, in, out, in, out) is semidet.
:- mode map_fold4_maybe(
    pred(in, out, in, out, in, out, in, out, mdi, muo) is semidet,
    in, out, in, out, in, out, in, out, mdi, muo) is semidet.
:- mode map_fold4_maybe(
    pred(in, out, in, out, in, out, in, out, di, uo) is semidet,
    in, out, in, out, in, out, in, out, di, uo) is semidet.
:- pred map_fold5_maybe(pred(T, U, A, A, B, B, C, C, D, D, E, E),
    maybe(T), maybe(U), A, A, B, B, C, C, D, D, E, E).
:- mode map_fold5_maybe(
    pred(in, out, in, out, in, out, in, out, in, out, in, out) is det,
    in, out, in, out, in, out, in, out, in, out, in, out) is det.
:- mode map_fold5_maybe(
    pred(in, out, in, out, in, out, in, out, in, out, mdi, muo) is det,
    in, out, in, out, in, out, in, out, in, out, mdi, muo) is det.
:- mode map_fold5_maybe(
    pred(in, out, in, out, in, out, in, out, in, out, di, uo) is det,
    in, out, in, out, in, out, in, out, in, out, di, uo) is det.
:- mode map_fold5_maybe(
    pred(in, out, in, out, in, out, in, out, in, out, in, out) is semidet,
    in, out, in, out, in, out, in, out, in, out, in, out) is semidet.
:- mode map_fold5_maybe(
    pred(in, out, in, out, in, out, in, out, in, out, mdi, muo) is semidet,
    in, out, in, out, in, out, in, out, in, out, mdi, muo) is semidet.
:- mode map_fold5_maybe(
    pred(in, out, in, out, in, out, in, out, in, out, di, uo) is semidet,
    in, out, in, out, in, out, in, out, in, out, di, uo) is semidet.
:- pred maybe_is_yes(maybe(T)::in, T::out) is semidet.
:- func pred_to_maybe(pred(T)) = maybe(T).
:- mode pred_to_maybe(pred(out) is semidet) = out is det.
:- func func_to_maybe((func) = T) = maybe(T).
:- mode func_to_maybe((func) = out is semidet) = out is det.
:- func maybe_default(T, maybe(T)) = T.

:- implementation.

map_maybe(_, no) = no.
map_maybe(F, yes(T)) = yes(F(T)).

map_maybe(_, no, no).
map_maybe(P, yes(T), yes(U)) :-
    P(T, U).

fold_maybe(_, no, A) = A.
fold_maybe(F, yes(T), A) = F(T, A).

fold_maybe(_, no, !A).
fold_maybe(P, yes(T), !A) :-
    P(T, !A).

fold2_maybe(_, no, !A, !B).
fold2_maybe(P, yes(T), !A, !B) :-
    P(T, !A, !B).

fold3_maybe(_, no, !A, !B, !C).
fold3_maybe(P, yes(T), !A, !B, !C) :-
    P(T, !A, !B, !C).

fold4_maybe(_, no, !A, !B, !C, !D).
fold4_maybe(P, yes(T), !A, !B, !C, !D) :-
    P(T, !A, !B, !C, !D).

fold5_maybe(_, no, !A, !B, !C, !D, !E).
fold5_maybe(P, yes(T), !A, !B, !C, !D, !E) :-
    P(T, !A, !B, !C, !D, !E).

map_fold_maybe(_, no, no, !A).
map_fold_maybe(P, yes(T), yes(U), !A) :-
    P(T, U, !A).

map_fold2_maybe(_, no, no, !A, !B).
map_fold2_maybe(P, yes(T), yes(U), !A, !B) :-
    P(T, U, !A, !B).

map_fold3_maybe(_, no, no, !A, !B, !C).
map_fold3_maybe(P, yes(T), yes(U), !A, !B, !C) :-
    P(T, U, !A, !B, !C).

map_fold4_maybe(_, no, no, !A, !B, !C, !D).
map_fold4_maybe(P, yes(T), yes(U), !A, !B, !C, !D) :-
    P(T, U, !A, !B, !C, !D).

map_fold5_maybe(_, no, no, !A, !B, !C, !D, !E).
map_fold5_maybe(P, yes(T), yes(U), !A, !B, !C, !D, !E) :-
    P(T, U, !A, !B, !C, !D, !E).

maybe_is_yes(yes(X), X).

pred_to_maybe(P) = ( if P(X) then yes(X) else no ).

func_to_maybe(F) = ( if X = apply(F) then yes(X) else no ).

maybe_default(Default, no) = Default.
maybe_default(_, yes(X)) = X.
