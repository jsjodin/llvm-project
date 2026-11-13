% A `ui` argument of a predicate of several clauses is as unique in each
% clause as in a predicate of one, and stays so: each clause's argument
% stands for the head variable, HeadVar__N, as mmc renames a head's
% variable to it. So a clause may pass it on to a `ui` argument, in a
% condition too, and a unique array to an `array_ui` one, as in the sound
% array stub; a clause that clobbers it is still an error. mmc unifies
% such an argument with its head variable and both are used after, so the
% argument is shared and every `ui` call is rejected (-fmmc-uniqueness;
% reference manual, "Limitations of the current implementation": "it is
% also not possible to use unique-input modes"). A head variable that is
% not unique keeps its own inst (`Semantics/modes/polymorphic-modes.m`).
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: %s < %t.err
% RUN: not merkc -fc1 -fdebug-dump-determinism -fmmc-uniqueness %s 2>%t.mmc >/dev/null
% RUN: FileCheck --check-prefix=MMC --implicit-check-not=error: %s < %t.mmc

:- module multi_clause_ui.
:- interface.
:- import_module array.
:- import_module list.
:- type box ---> box(int).
:- pred count(list(int)::in, box::ui, int::in, int::out) is det.
:- pred zeros(list(int)::in, array(int)::array_ui, int::in, int::out) is det.
:- pred clobbers(list(int)::in, box::ui, box::uo) is det.
:- implementation.
:- import_module int.

count([], _, !S).
count([X | Xs], B, !S) :-
    ( if peek(B) = X then
        !:S = !.S + 1
    else
        true
    ),
    count(Xs, B, !S).
% MMC: multi-clause-ui.m:[[@LINE-6]]:10: error: in the clause for count(in, ui, in, out): mode error in argument 1 of the call to func multi_clause_ui.peek/1: B has inst ground, but unique is expected

zeros([], _, !S).
zeros([X | Xs], A, !S) :-
    ( if A ^ unsafe_elem(X) = 0 then
        !:S = !.S + 1
    else
        true
    ),
    zeros(Xs, A, !S).

clobbers([], B, C) :-
    C = copy_box(B).
clobbers([_ | _], B, C) :-
    eat(B, C).
% ERR: multi-clause-ui.m:[[@LINE-2]]:22: error: in the clause for clobbers(in, ui, uo): mode error: argument 2, HeadVar__2, did not get sufficiently instantiated: its final inst is clobbered, but unique is expected
% MMC: multi-clause-ui.m:[[@LINE-4]]:9: error: in the clause for clobbers(in, ui, uo): mode error in argument 1 of the call to func multi_clause_ui.copy_box/1: B has inst ground, but unique is expected

    % Reading a `ui` term's field: mmc makes the term `unique(box(ground))`,
    % which it then says is not `unique` (Semantics/uniqueness/copied.m).
:- func peek(box::ui) = (int::out) is det.
peek(B) = N :-
    B = box(N).
% MMC: multi-clause-ui.m:[[@LINE-1]]:9: error: in the clause for peek(ui) = out: mode error: argument 1, B, did not get sufficiently instantiated: its final inst is unique(box(ground)), but unique is expected

:- func copy_box(box::ui) = (box::uo) is det.
copy_box(B) = box(N) :-
    B = box(N).
% MMC: multi-clause-ui.m:[[@LINE-2]]:15: error: in the clause for copy_box(ui) = uo: mode error: argument 1, B, did not get sufficiently instantiated: its final inst is unique(box(ground)), but unique is expected

:- pred eat(box::di, box::uo) is det.
eat(box(N), box(N + 1)).
