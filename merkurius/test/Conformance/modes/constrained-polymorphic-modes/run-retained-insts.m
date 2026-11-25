% MANUAL: Constrained polymorphic modes
% An inst parameter constrained by I =< Inst stands for any inst
% compatible with Inst, and a call's result keeps the caller's inst: a
% list of closures passed through a polymorphic predicate can still be
% called. The constraint may follow the declaration after <=, and a ground
% constraint may be left out.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:11
% OUTPUT-NEXT:7
% OUTPUT-NEXT:3
% OUTPUT-NOT:{{.}}

:- module run_retained_insts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pred app(list(T), list(T), list(T)).
:- mode app(in(list_skel(I =< ground)), in(list_skel(I =< ground)),
    out(list_skel(I =< ground))) is det.
app([], Ys, Ys).
app([X | Xs], Ys, [X | Zs]) :-
    app(Xs, Ys, Zs).

:- pred first(list(T), T).
:- (mode first(in(list_skel(I)), out(I)) is semidet) <= I =< ground.
first([X | _], X).

:- pred last_of(list(T), T).
:- mode last_of(in(list_skel(I)), out(I)) is semidet.
last_of([X], X).
last_of([_, Y | Ys], Z) :-
    last_of([Y | Ys], Z).

:- inst adder == (func(in) = out is det).

main(!IO) :-
    Fs0 = [(func(X) = X + 1), (func(X) = X * 2)],
    Gs0 = [(func(X) = X - 3)],
    app(Fs0, Gs0, Fs),
    ( if first(Fs, F) then
        io.write_line(apply(F, 10), !IO)
    else
        io.write_string("empty\n", !IO)
    ),
    ( if last_of(Fs, G) then
        io.write_line(apply(G, 10), !IO)
    else
        io.write_string("empty\n", !IO)
    ),
    io.write_line(list.length(Fs) : int, !IO).
