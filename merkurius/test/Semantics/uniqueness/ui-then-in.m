% A term built in the clause becomes shared when it is passed to an `in`
% argument, which its callee may keep in an output, whether or not it was
% passed to a `ui` argument before. mmc keeps it unique after a `ui` call,
% and accepts `after_ui`, though L holds T (see the design document's
% register); so this is not compared with mmc, and merkc rejects both
% with and without -fmmc-uniqueness.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: %s < %t.err
% RUN: not merkc -fc1 -fdebug-dump-determinism -fmmc-uniqueness %s 2>%t.mmc.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: %s < %t.mmc.err

:- module ui_then_in.
:- interface.
:- import_module list.
:- type t ---> t(int).
:- pred after_ui(t::uo, list(t)::out) is det.
:- pred no_ui(t::uo, list(t)::out) is det.
:- implementation.
:- pred ui(t::ui) is det.
ui(_).
:- pred wrap(t::in, list(t)::out) is det.
wrap(X, [X]).
after_ui(T, L) :-
    T0 = t(1), ui(T0), wrap(T0, L),
    T = T0.
% ERR: ui-then-in.m:[[@LINE-1]]:5: error: in the clause for after_ui(uo, out): mode error: argument 1, T, had the wrong inst: its final inst is bound(t(bound(1))), but unique is expected
no_ui(T, L) :-
    T0 = t(1), wrap(T0, L),
    T = T0.
% ERR: ui-then-in.m:[[@LINE-1]]:5: error: in the clause for no_ui(uo, out): mode error: argument 1, T, had the wrong inst: its final inst is bound(t(bound(1))), but unique is expected
