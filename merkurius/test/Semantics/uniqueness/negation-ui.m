% A negated goal must not clobber a value that is used after the negation,
% as it is needed when the goal fails, nor a value made from it; but a use
% that clobbers nothing, as a `ui` argument, leaves it unique. Before, every
% value used after a negation was mostly_unique inside it, so a condition
% `A ^ unsafe_elem(I) \= 0`, with `A::array_ui` and A used after the
% if-then-else, was an error that mmc does not give (found writing
% benchmarks/life/life-places.m). mmc rejects a clobber in a negation of a
% value of most types, but not of an array, and the array then holds the
% negated goal's update (see the design document's register, "An array
% clobbered in a negation"): clobbered_in_condition, which mmc accepts.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: %s < %t.err
% RUN: not merkc -fc1 -fdebug-dump-determinism -fmmc-uniqueness %s 2>%t.mmc >/dev/null
% RUN: FileCheck --check-prefix=MMC --implicit-check-not=error: %s < %t.mmc

:- module negation_ui.
:- interface.
:- import_module array.
:- import_module list.
:- type t ---> t(int).
:- pred count(array(int)::array_ui, list(int)::in, int::in, int::out) is det.
:- pred count_ne(array(int)::array_ui, list(int)::in, int::in, int::out)
    is det.
:- pred next_after(array(int)::array_ui, int::in, int::out) is semidet.
:- pred clobbered_in_condition(array(int)::array_di, array(int)::array_uo,
    int::out) is det.
:- pred clobbered_alias(t::di, t::uo) is semidet.
:- implementation.
:- import_module int.

count(_, [], !N).
count(A, [I | Is], !N) :-
    ( if not A ^ unsafe_elem(I) = 0 then
        !:N = !.N + 1
    else
        true
    ),
    count(A, Is, !N).

count_ne(_, [], !N).
count_ne(A, [I | Is], !N) :-
    ( if A ^ unsafe_elem(I) \= 0 then
        !:N = !.N + 1
    else
        true
    ),
    count_ne(A, Is, !N).

next_after(A, I, J) :-
    not A ^ unsafe_elem(I) = 0,
    J = A ^ unsafe_elem(I + 1).

clobbered_in_condition(A0, A, N) :-
    ( if not (A1 = A0 ^ elem(0) := 7, A1 ^ elem(0) = 3) then
        N = 1
    else
        N = 0
    ),
    A = A0.
% ERR: negation-ui.m:[[@LINE-6]]:20: error: in the clause for clobbered_in_condition(di(array.uniq_array), out(array.uniq_array), out): mode error in argument 2 of the call to func array.elem :=/3: A0 has inst mostly_unique, but array.uniq_array is expected

clobbered_alias(T0, T) :-
    not (T1 = T0, updu(1, T1, T2), T2 = t(3)),
    updu(2, T0, T).
% ERR: negation-ui.m:[[@LINE-2]]:19: error: in the clause for clobbered_alias(di, uo): mode error in argument 2 of the call to pred negation_ui.updu/3: T1 has inst mostly_unique, but unique is expected
% MMC: negation-ui.m:[[@LINE-3]]:19: error: in the clause for clobbered_alias(di, uo): mode error in argument 2 of the call to pred negation_ui.updu/3: T1 has inst mostly_unique, but unique is expected

:- pred updu(int::in, t::di, t::uo) is det.
updu(Y, t(X), t(X + Y)).
