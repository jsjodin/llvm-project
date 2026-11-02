% Unique arrays: by default merkc loads the sound array stub, in which
% arrays are unique and the library's predicates that read one keep it
% unique (array_ui); with -fmmc-uniqueness, the library manual's stub, in
% which arrays are ground, as in mmc. Filling, summing, swapping, field
% syntax, sorting, folds over an index or the array, closures, committed
% choices, if-then-elses, a sieve, a histogram, a bubble sort and a grid
% are accepted either way. By default an array cannot be updated after a
% predicate of the program's own has read it `in`, which shares it, or
% after a choice point, where mmc's updates are seen by every solution.
% Either way, a mostly unique array cannot be updated by array.set, which
% clobbers it. By default, too, a closure over a unique array may be folded with
% array.foldl, and a term holding a unique array and a number is unique.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: %s < %t.err
% RUN: not merkc -fc1 -fdebug-dump-determinism -fmmc-uniqueness %s 2>%t.mmc >/dev/null
% RUN: FileCheck --check-prefix=MMC --implicit-check-not=error: %s < %t.mmc
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module arrays.
:- interface.
:- import_module array, int, list.

:- pred fill_down(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred sum(array(int)::array_ui, int::out) is det.
:- pred sum_in(array(int)::in, int::out) is det.
:- pred swap(int::in, int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred field_syntax(array(int)::array_di, array(int)::array_uo) is det.
:- pred bump_all(array(int)::array_di, array(int)::array_uo) is det.
:- pred from_list(list(int)::in, array(int)::array_uo) is det.
:- pred sorted(array(int)::array_di, array(int)::array_uo) is det.
:- pred maybe_set(int::in, array(int)::array_di, array(int)::array_uo) is semidet.
:- pred mostly(array(int)::array_mdi, array(int)::array_muo) is nondet.
:- pred own_reader_then_set(array(int)::array_di, array(int)::array_uo) is det.
:- pred ui_reader_then_set(array(int)::array_di, array(int)::array_uo) is det.
:- pred kept_copy(array(int)::array_di, array(int)::array_uo, array(int)::out) is det.
:- pred in_term(array(int)::array_di, {array(int), int}::out) is det.
:- pred updated_after_choice(array(int)::array_di, array(int)::array_uo) is multi.
:- pred generated(int::in, array(int)::array_uo) is det.
:- pred threaded(int::in, int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred up(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred mapped(array(int)::array_di, array(int)::array_uo) is det.
:- pred folded(array(int)::array_ui, int::out) is det.
:- pred folded_state(array(int)::array_ui, array(int)::array_di, array(int)::array_uo) is det.
:- pred lambda_set(array(int)::array_di, array(int)::array_uo) is det.
:- pred closure_set(array(int)::array_di, array(int)::array_uo) is det.
:- pred mostly_fold(int::in, array(int)::array_mdi, array(int)::array_muo) is det.
:- pred committed(array(int)::array_di, array(int)::array_uo) is cc_multi.
:- pred chosen(array(int)::array_di, array(int)::array_uo) is cc_multi.
:- pred member_then_set(array(int)::array_di, array(int)::array_uo) is cc_nondet.
:- pred cond_set(array(int)::array_di, array(int)::array_uo) is det.
:- pred cond_reads(array(int)::array_di, array(int)::array_uo) is det.
:- pred two(array(int)::array_di, array(int)::array_di, array(int)::array_uo) is det.
:- pred same_twice(array(int)::array_di, array(int)::array_uo) is det.
:- pred to_list_then_set(array(int)::array_di, array(int)::array_uo, list(int)::out) is det.
:- pred resized(array(int)::array_di, array(int)::array_uo) is det.
:- pred semidet_lookup_then_set(array(int)::array_di, array(int)::array_uo) is semidet.
:- pred sieve(int::in, list(int)::out) is det.
:- pred histogram(list(int)::in, int::in, array(int)::array_uo) is det.
:- pred bubble(array(int)::array_di, array(int)::array_uo) is det.
:- pred bubble_in(array(int)::array_di, array(int)::array_uo) is det.
:- type grid ---> grid(int, array(int)).
:- pred grid_set(int::in, int::in, int::in, grid::di, grid::uo) is det.
:- pred grid_get(grid::in, int::in, int::in, int::out) is det.
:- pred grid_bump(int::in, int::in, grid::di, grid::uo) is det.

:- implementation.

fill_down(N, !A) :- ( if N > 0 then array.set(N - 1, N, !A), fill_down(N - 1, !A) else true ).
sum(A, S) :- S = array.foldl(func(X, S0) = X + S0, A, 0).
sum_in(A, S) :- array.size(A, S).
swap(I, J, !A) :- X = array.lookup(!.A, I), Y = array.lookup(!.A, J), array.set(I, Y, !A), array.set(J, X, !A).
field_syntax(!A) :- X = !.A ^ elem(0), !A ^ elem(1) := X.
bump_all(!A) :- array.bounds(!.A, Lo, Hi), bump(Lo, Hi, !A).
:- pred bump(int::in, int::in, array(int)::array_di, array(int)::array_uo) is det.
bump(I, Hi, !A) :- ( if I =< Hi then X = !.A ^ elem(I), !A ^ elem(I) := X + 1, bump(I + 1, Hi, !A) else true ).
from_list(L, A) :- A = array.from_list(L).
sorted(!A) :- !:A = array.sort(!.A).
maybe_set(I, !A) :- array.semidet_set(I, 0, !A).
mostly(!A) :- ( X = 1 ; X = 2 ), array.set(0, X, !A).
% ERR-DAG: arrays.m:[[@LINE-1]]:34: error: in the clause for mostly((array.mostly_uniq_array >> mostly_clobbered), out(array.mostly_uniq_array)): mode error in argument 3 of the call to pred array.set/4: A!0 has inst array.mostly_uniq_array, but array.uniq_array is expected
% MMC-DAG: arrays.m:[[@LINE-2]]:34: error: in the clause for mostly((array.mostly_uniq_array >> mostly_clobbered), out(array.mostly_uniq_array)): mode error: argument 1, A!0, did not get sufficiently instantiated: its final inst is clobbered, but mostly_clobbered is expected
own_reader_then_set(!A) :- sum_in(!.A, S), array.set(0, S, !A).
% ERR-DAG: arrays.m:[[@LINE-1]]:44: error: in the clause for own_reader_then_set(di(array.uniq_array), out(array.uniq_array)): mode error in argument 3 of the call to pred array.set/4: A!0 has inst ground, but array.uniq_array is expected
ui_reader_then_set(!A) :- sum(!.A, S), array.set(0, S, !A).
kept_copy(A0, A, C) :- C = array.copy(A0), array.set(0, 1, A0, A).
in_term(A, {A, 1}).
updated_after_choice(!A) :- ( X = 1 ; X = 2 ), array.set(0, X, !A).
% ERR-DAG: arrays.m:[[@LINE-1]]:48: error: in the clause for updated_after_choice(di(array.uniq_array), out(array.uniq_array)): mode error in argument 3 of the call to pred array.set/4: STATE_VARIABLE_A_0 has inst mostly_unique, but array.uniq_array is expected
generated(N, A) :- A = array.generate(N, func(I) = I * 2).
threaded(I, X, !A) :- array.set(I, X, !A), Y = array.lookup(!.A, I), array.set(I + 1, Y, !A).
up(N, !A) :- int.fold_up(pred(I::in, B0::array_di, B::array_uo) is det :- array.set(I, I, B0, B), 0, N - 1, !A).
mapped(!A) :- !:A = array.map(func(X) = X + 1, !.A).
folded(A, S) :- array.foldl(pred(X::in, S0::in, S1::out) is det :- S1 = S0 + X, A, 0, S).
folded_state(R, !A) :- array.foldl(pred(X::in, B0::array_di, B::array_uo) is det :- array.set(0, X, B0, B), R, !A).
% MMC-DAG: arrays.m:[[@LINE-1]]:24: error: in the clause for folded_state(in(array.uniq_array), di(array.uniq_array), out(array.uniq_array)): mode error in the call to pred array.foldl/4: the arguments V_2, R, A!0, A!1 have insts /* unique */ (pred(in, di(array.uniq_array), out(array.uniq_array)) is det), array.uniq_array, array.uniq_array, free, which match none of its modes
lambda_set(!A) :- P = (pred(B0::array_di, B::array_uo) is det :- array.set(0, 1, B0, B)), P(!A).
closure_set(!A) :- P = array.set(0, 1), P(!A).
mostly_fold(N, !A) :- int.fold_up(pred(I::in, B0::array_mdi, B::array_muo) is det :- B = B0, 0, N, !A).
% MMC-DAG: arrays.m:[[@LINE-1]]:23: error: in the clause for mostly_fold(in, (array.mostly_uniq_array >> mostly_clobbered), out(array.mostly_uniq_array)): mode error: argument 2, A!0, did not get sufficiently instantiated: its final inst is clobbered, but mostly_clobbered is expected
committed(!A) :- ( X = 1 ; X = 2 ), array.set(0, X, !A).
chosen(!A) :- promise_equivalent_solutions [X] ( X = 1 ; X = 2 ), array.set(0, X, !A).
member_then_set(!A) :- array.member(!.A, X), array.set(0, X, !A).
cond_set(!A) :- ( if array.semidet_lookup(!.A, 3, X), X > 0 then array.set(0, X, !A) else true ).
cond_reads(!A) :- ( if array.lookup(!.A, 0) > 0 then array.set(0, 0, !A) else array.set(1, 0, !A) ).
two(A, !B) :- array.set(0, array.lookup(A, 0), !B).
same_twice(A0, A) :- array.set(0, 1, A0, A1), array.set(1, 2, A0, A2), A = array.append(A1, A2).
% ERR-DAG: arrays.m:[[@LINE-1]]:22: error: in the clause for same_twice(di(array.uniq_array), out(array.uniq_array)): unique-mode error in argument 3 of the call to pred array.set/4: the called procedure would clobber its argument, but A0 is still live
% MMC-DAG: arrays.m:[[@LINE-2]]:22: error: in the clause for same_twice(di(array.uniq_array), out(array.uniq_array)): unique-mode error in argument 3 of the call to pred array.set/4: the called procedure would clobber its argument, but A0 is still live
to_list_then_set(!A, L) :- L = array.to_list(!.A), array.set(0, 1, !A).
resized(!A) :- array.resize(10, 0, !A).
semidet_lookup_then_set(!A) :- array.semidet_lookup(!.A, 0, X), array.set(1, X, !A).
sieve(N, Ps) :-
    A0 = array.init(N + 1, yes),
    cross(2, N, A0, A),
    collect(2, N, A, [], Ps0),
    list.reverse(Ps0, Ps).
:- type yes_no ---> yes ; no.
:- pred cross(int::in, int::in, array(yes_no)::array_di, array(yes_no)::array_uo) is det.
cross(I, N, !A) :-
    ( if I * I =< N then
        ( if array.lookup(!.A, I) = yes then strike(I * I, I, N, !A) else true ),
        cross(I + 1, N, !A)
    else true ).
:- pred strike(int::in, int::in, int::in, array(yes_no)::array_di, array(yes_no)::array_uo) is det.
strike(J, I, N, !A) :- ( if J =< N then array.set(J, no, !A), strike(J + I, I, N, !A) else true ).
:- pred collect(int::in, int::in, array(yes_no)::in, list(int)::in, list(int)::out) is det.
collect(I, N, A, !Ps) :-
    ( if I =< N then
        ( if array.lookup(A, I) = yes then !:Ps = [I | !.Ps] else true ),
        collect(I + 1, N, A, !Ps)
    else true ).
histogram(Xs, N, H) :- H0 = array.init(N, 0), count(Xs, H0, H).
:- pred count(list(int)::in, array(int)::array_di, array(int)::array_uo) is det.
count([], !H).
count([X | Xs], !H) :- C = array.lookup(!.H, X), array.set(X, C + 1, !H), count(Xs, !H).
bubble(!A) :- array.size(!.A, N), pass(0, N - 1, !A).
:- pred pass(int::in, int::in, array(int)::array_di, array(int)::array_uo) is det.
pass(I, Last, !A) :-
    ( if I < Last then
        X = !.A ^ elem(I), Y = !.A ^ elem(I + 1),
        ( if X > Y then !A ^ elem(I) := Y, !A ^ elem(I + 1) := X else true ),
        pass(I + 1, Last, !A)
    else true ).
bubble_in(!A) :- length_of(!.A, N), pass(0, N - 1, !A).
% ERR-DAG: arrays.m:[[@LINE-1]]:37: error: in the clause for bubble_in(di(array.uniq_array), out(array.uniq_array)): mode error in argument 3 of the call to pred arrays.pass/4: A!0 has inst ground, but array.uniq_array is expected
:- pred length_of(array(int)::in, int::out) is det.
length_of(A, N) :- array.size(A, N).
grid_set(R, C, X, grid(W, A0), grid(W, A)) :- array.set(R * W + C, X, A0, A).
% MMC-DAG: arrays.m:[[@LINE-1]]:32: error: in the clause for grid_set(in, in, in, di, uo): mode error: argument 5, HeadVar__5, did not get sufficiently instantiated: its final inst is unique(grid(ground, array.uniq_array)), but unique is expected
grid_get(grid(W, A), R, C, X) :- X = array.lookup(A, R * W + C).
grid_bump(R, C, G0, G) :- grid_get(G0, R, C, X), grid_set(R, C, X + 1, G0, G).
% ERR-DAG: arrays.m:[[@LINE-1]]:50: error: in the clause for grid_bump(in, in, di, uo): mode error in argument 4 of the call to pred arrays.grid_set/5: G0 has inst ground, but unique is expected
% MMC-DAG: arrays.m:[[@LINE-2]]:50: error: in the clause for grid_bump(in, in, di, uo): mode error in argument 4 of the call to pred arrays.grid_set/5: G0 has inst ground, but unique is expected
