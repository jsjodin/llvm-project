% A mode inferred from a call with a unique argument not used after is
% unique, so di, as in mmc. Where uniqueness checking then finds that
% argument mostly unique at a call of the procedure, as at a recursive call
% after a choice point, the mode is inferred again with it shared, so in:
% a loop over a literal list whose body has several solutions. mmc infers
% the unique mode only, and aborts on most such programs ("found detism
% error"). An argument that stays unique, as an io state, keeps di.
%
% RUN: merkc -fc1 -fdo-loops -fdebug-dump-determinism %s -o - | FileCheck %s --check-prefix=MODES
% RUN: merkc -O0 -fdo-loops %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module inferred_shared.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions.

:- pred choices(list(int)::out) is multi.
choices(L) :- mloop([1], L).

mloop([], []).
mloop([X | Xs], [Y | Ys]) :- ( Y = X ; Y = X * 10 ), mloop(Xs, Ys).
% MODES-DAG: pred inferred_shared.mloop/2 inferred mode 0: mloop(in, out), inferred multi

:- pred loop_choices(list(int)::out) is multi.
loop_choices(L) :-
    ( foreach(X, [1, 2]), foreach(Y, L) do ( Y = X ; Y = X * 10 ) ).
% MODES-DAG: pred inferred_shared.loop_choices__loop_[[@LINE-1]]_7/2 inferred mode 0: loop_choices__loop_[[@LINE-1]]_7(in, out), inferred multi

:- pred doubled(list(int)::out) is det.
doubled(L) :- dloop([1, 2], L).

dloop([], []).
dloop([X | Xs], [X * 2 | Ys]) :- dloop(Xs, Ys).
% MODES-DAG: pred inferred_shared.dloop/2 inferred mode 0: dloop(di, uo), inferred det

say(S, !IO) :- io.write_string(S, !IO), io.nl(!IO).
% MODES-DAG: pred inferred_shared.say/3 inferred mode 0: say(di, di, uo), inferred det

main(!IO) :-
    solutions(choices, A), io.write(A, !IO), io.nl(!IO),
    solutions(loop_choices, B), io.write(B, !IO), io.nl(!IO),
    doubled(C), io.write(C, !IO), io.nl(!IO),
    say("done", !IO).
% OUTPUT:{{\[}}[1], [10]]
% OUTPUT-NEXT:{{\[}}[1, 2], [1, 20], [10, 2], [10, 20]]
% OUTPUT-NEXT:[2, 4]
% OUTPUT-NEXT:done
