% Unique-mode errors of the first phase (Overton, "Precise and Expressive
% Mode Systems for Typed Logic Programming Languages", 2003, sections
% 3.2.1 and 3.2.3; reference manual, "Unique modes"): a call that would
% clobber a variable used after, first-order or higher-order, which mmc
% reports before an inst that does not match; a value shared by an
% assignment, by an `in' argument, by a test of two variables used after,
% or as a part that both the term and its variable use after, then passed
% where a unique one is expected; and a uo output that is only ground,
% which mmc says "did not get sufficiently instantiated". mostly_unique
% values, and those needed on backtracking, are for the second phase,
% after determinism analysis.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=because: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module uniqueness_errors.
:- interface.
:- import_module io, list.

:- pred twice(io::di, io::uo) is det.
:- pred higher_order(io::di, io::uo) is det.
:- pred assigned is det.
:- pred passed_in(list(int)::uo, list(list(int))::out) is det.
:- pred tested(list(int)::di, list(int)::di) is semidet.
:- pred part(list(list(int))::di) is semidet.
:- pred ground_live(list(int)::in, list(int)::out) is det.
:- pred only_ground(list(int)::di, list(int)::uo) is det.

:- implementation.

:- pred d(list(int)::di) is det.
d(_).
:- pred dd(list(int)::di, list(int)::uo) is det.
dd(X, X).
:- pred use(list(int)::in) is det.
use(_).
:- pred usel(list(list(int))::in) is det.
usel(_).

twice(IO0, IO) :-
    io.write_int(1, IO0, _),
% CHECK: [[@LINE-1]]:5: error: in the clause for twice(di, uo): unique-mode error in argument 2 of the call to pred io.write_int/3: the called procedure would clobber its argument, but IO0 is still live
    io.write_int(2, IO0, IO).

higher_order(IO0, IO) :-
    P = (pred(S0::di, S::uo) is det :- io.write_int(1, S0, S)),
    P(IO0, _),
% CHECK: [[@LINE-1]]:5: error: in the clause for higher_order(di, uo): unique-mode error in argument 2 of the higher-order predicate call: the called procedure would clobber its argument, but IO0 is still live
    P(IO0, IO).

assigned :-
    X = [1],
    Y = X,
    d(X),
% CHECK: [[@LINE-1]]:5: error: in the clause for assigned: mode error in argument 1 of the call to pred uniqueness_errors.d/1: X has inst bound('[|]'(bound(1), bound([]))), but unique is expected
    use(Y).

passed_in(X, L) :-
    X = [1],
    L = [X],
    use(X).
% CHECK: [[@LINE-1]]:5: error: in the clause for passed_in(uo, out): mode error: argument 1, X, had the wrong inst: its final inst is bound('[|]'(bound(1), bound([]))), but unique is expected

tested(X, Y) :-
    X = Y,
    d(X),
% CHECK: [[@LINE-1]]:5: error: in the clause for tested(di, di): mode error in argument 1 of the call to pred uniqueness_errors.d/1: X has inst ground, but unique is expected
    use(Y).

part(L) :-
    L = [X | _],
    d(X),
% CHECK: [[@LINE-1]]:5: error: in the clause for part(di): mode error in argument 1 of the call to pred uniqueness_errors.d/1: X has inst ground, but unique is expected
    usel(L).

ground_live(X, Y) :-
    d(X),
% CHECK: [[@LINE-1]]:5: error: in the clause for ground_live(in, out): unique-mode error in argument 1 of the call to pred uniqueness_errors.d/1: the called procedure would clobber its argument, but X is still live
    Y = X.

only_ground(X, Y) :-
    dd(X, Z),
    Y = Z,
    use(Z).
% CHECK: [[@LINE-1]]:5: error: in the clause for only_ground(di, uo): mode error: argument 2, Y, did not get sufficiently instantiated: its final inst is ground, but unique is expected
