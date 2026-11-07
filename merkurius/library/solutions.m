% The standard library module solutions, with the declarations of the
% library reference manual (22.01), less solutions_set/1 and
% solutions_set/2, which need the module set, not written yet; see
% README.md. unsorted_aggregate/4, unsorted_aggregate2/6 and do_while/4
% have no clauses: code generation makes each of their procedures from
% merk.aggregate (design document, Lowering, "Collecting solutions"). The
% others are written in Mercury on them.

:- module solutions.
:- interface.
:- import_module bool.
:- import_module list.

:- pred solutions(pred(T), list(T)).
:- mode solutions(pred(out) is multi, out(non_empty_list)) is det.
:- mode solutions(pred(out) is nondet, out) is det.
:- func solutions(pred(T)) = list(T).
:- mode solutions(pred(out) is multi) = out(non_empty_list) is det.
:- mode solutions(pred(out) is nondet) = out is det.
:- pred unsorted_solutions(pred(T), list(T)).
:- mode unsorted_solutions(pred(out) is multi, out(non_empty_list))
    is cc_multi.
:- mode unsorted_solutions(pred(out) is nondet, out)
    is cc_multi.
:- func aggregate(pred(T), func(T, U) = U, U) = U.
:- mode aggregate(pred(out) is multi, func(in, in) = out is det, in)
    = out is det.
:- mode aggregate(pred(out) is nondet, func(in, in) = out is det, in)
    = out is det.
:- pred aggregate(pred(T), pred(T, U, U), U, U).
:- mode aggregate(pred(out) is multi, pred(in, in, out) is det,
    in, out) is det.
:- mode aggregate(pred(out) is multi, pred(in, di, uo) is det,
    di, uo) is det.
:- mode aggregate(pred(out) is nondet, pred(in, di, uo) is det,
    di, uo) is det.
:- mode aggregate(pred(out) is nondet, pred(in, in, out) is det,
    in, out) is det.
:- pred aggregate2(pred(T), pred(T, U, U, V, V), U, U, V, V).
:- mode aggregate2(pred(out) is multi, pred(in, in, out, in, out) is det,
    in, out, in, out) is det.
:- mode aggregate2(pred(out) is multi, pred(in, in, out, di, uo) is det,
    in, out, di, uo) is det.
:- mode aggregate2(pred(out) is nondet, pred(in, in, out, di, uo) is det,
    in, out, di, uo) is det.
:- mode aggregate2(pred(out) is nondet, pred(in, in, out, in, out) is det,
    in, out, in, out) is det.
:- pred unsorted_aggregate(pred(T), pred(T, U, U), U, U).
:- mode unsorted_aggregate(pred(out) is multi, pred(in, in, out) is det,
    in, out) is cc_multi.
:- mode unsorted_aggregate(pred(out) is multi, pred(in, in, out) is cc_multi,
    in, out) is cc_multi.
:- mode unsorted_aggregate(pred(out) is multi, pred(in, di, uo) is det,
    di, uo) is cc_multi.
:- mode unsorted_aggregate(pred(out) is multi, pred(in, di, uo) is cc_multi,
    di, uo) is cc_multi.
:- mode unsorted_aggregate(pred(muo) is multi, pred(mdi, di, uo) is det,
    di, uo) is cc_multi.
:- mode unsorted_aggregate(pred(out) is nondet, pred(in, di, uo) is det,
    di, uo) is cc_multi.
:- mode unsorted_aggregate(pred(out) is nondet, pred(in, di, uo) is cc_multi,
    di, uo) is cc_multi.
:- mode unsorted_aggregate(pred(out) is nondet, pred(in, in, out) is det,
    in, out) is cc_multi.
:- mode unsorted_aggregate(pred(out) is nondet, pred(in, in, out) is cc_multi,
    in, out) is cc_multi.
:- mode unsorted_aggregate(pred(muo) is nondet, pred(mdi, di, uo) is det,
    di, uo) is cc_multi.
:- pred unsorted_aggregate2(pred(T), pred(T, U, U, V, V), U, U, V, V).
:- mode unsorted_aggregate2(pred(out) is multi,
    pred(in, in, out, in, out) is det, in, out, in, out) is cc_multi.
:- mode unsorted_aggregate2(pred(out) is multi,
    pred(in, in, out, in, out) is cc_multi, in, out, in, out) is cc_multi.
:- mode unsorted_aggregate2(pred(out) is multi,
    pred(in, in, out, di, uo) is det, in, out, di, uo) is cc_multi.
:- mode unsorted_aggregate2(pred(out) is multi,
    pred(in, in, out, di, uo) is cc_multi, in, out, di, uo) is cc_multi.
:- mode unsorted_aggregate2(pred(out) is nondet,
    pred(in, in, out, in, out) is det, in, out, in, out) is cc_multi.
:- mode unsorted_aggregate2(pred(out) is nondet,
    pred(in, in, out, in, out) is cc_multi, in, out, in, out) is cc_multi.
:- mode unsorted_aggregate2(pred(out) is nondet,
    pred(in, in, out, di, uo) is det, in, out, di, uo) is cc_multi.
:- mode unsorted_aggregate2(pred(out) is nondet,
    pred(in, in, out, di, uo) is cc_multi, in, out, di, uo) is cc_multi.
:- pred do_while(pred(T), pred(T, bool, T2, T2), T2, T2).
:- mode do_while(pred(out) is multi, pred(in, out, in, out) is det,
    in, out) is cc_multi.
:- mode do_while(pred(out) is multi, pred(in, out, di, uo) is det,
    di, uo) is cc_multi.
:- mode do_while(pred(out) is multi, pred(in, out, di, uo) is cc_multi,
    di, uo) is cc_multi.
:- mode do_while(pred(out) is nondet, pred(in, out, in, out) is det,
    in, out) is cc_multi.
:- mode do_while(pred(out) is nondet, pred(in, out, di, uo) is det,
    di, uo) is cc_multi.
:- mode do_while(pred(out) is nondet, pred(in, out, di, uo) is cc_multi,
    di, uo) is cc_multi.

:- implementation.
:- import_module require.

% The multi modes promise a non-empty list, which mode analysis does not
% see through the fold, so they have their own clauses. The list is in the
% reverse of the order the generator gives the solutions in, as mmc's is.
:- pragma promise_equivalent_clauses(pred(unsorted_solutions/2)).
unsorted_solutions(Generator::in(pred(out) is multi),
        List::out(non_empty_list)) :-
    unsorted_aggregate(Generator, list.cons, [], List0),
    non_empty(List0, List).
unsorted_solutions(Generator::in(pred(out) is nondet), List::out) :-
    unsorted_aggregate(Generator, list.cons, [], List).

% The sorted list is the same whatever order the solutions come in.
:- pragma promise_equivalent_clauses(pred(solutions/2)).
solutions(Generator::in(pred(out) is multi), List::out(non_empty_list)) :-
    promise_equivalent_solutions [List] (
        unsorted_solutions(Generator, List0),
        list.sort_and_remove_dups(List0, List1),
        non_empty(List1, List)
    ).
solutions(Generator::in(pred(out) is nondet), List::out) :-
    promise_equivalent_solutions [List] (
        unsorted_solutions(Generator, List0),
        list.sort_and_remove_dups(List0, List)
    ).

:- pragma promise_equivalent_clauses(func(solutions/1)).
solutions(Generator::in(pred(out) is multi)) =
        (List::out(non_empty_list)) :-
    solutions(Generator, List).
solutions(Generator::in(pred(out) is nondet)) = (List::out) :-
    solutions(Generator, List).

aggregate(Generator, Step, Acc0) =
    list.foldl(Step, solutions(Generator), Acc0).

aggregate(Generator, Step, !Acc) :-
    solutions(Generator, Solutions),
    list.foldl(Step, Solutions, !Acc).

aggregate2(Generator, Step, !Acc1, !Acc2) :-
    solutions(Generator, Solutions),
    list.foldl2(Step, Solutions, !Acc1, !Acc2).

:- pred non_empty(list(T)::in, list(T)::out(non_empty_list)) is det.

non_empty([], _) :-
    unexpected("solutions", "no solutions of a multi generator").
non_empty([X | Xs], [X | Xs]).
