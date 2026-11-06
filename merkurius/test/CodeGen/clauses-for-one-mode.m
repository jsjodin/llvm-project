% Clauses for one mode (reference manual, "Different clauses for different
% modes"): each procedure is generated from the clauses for its mode and
% those for every mode, and one that no clause is for fails. The output is
% mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s 2>/dev/null | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 2>/dev/null && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 2>/dev/null && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module clauses_for_one_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module int, list.

% The reference manual's example: append(out, in, in) has at most one
% solution, which the usual clauses do not show.
:- pred app(list(T), list(T), list(T)).
:- mode app(in, in, out) is det.
:- mode app(out, out, in) is multi.
:- mode app(in, out, in) is semidet.
:- mode app(out, in, in) is semidet.
:- pragma promise_equivalent_clauses(app/3).
app(L1::in, L2::in, L3::out) :- usual_app(L1, L2, L3).
app(L1::out, L2::out, L3::in) :- usual_app(L1, L2, L3).
app(L1::in, L2::out, L3::in) :- usual_app(L1, L2, L3).
app(L1::out, L2::in, L3::in) :- other_app(L1, L2, L3).

:- pred usual_app(list(T), list(T), list(T)).
:- mode usual_app(in, in, out) is det.
:- mode usual_app(out, out, in) is multi.
:- mode usual_app(in, out, in) is semidet.
usual_app([], Ys, Ys).
usual_app([X | Xs], Ys, [X | Zs]) :- usual_app(Xs, Ys, Zs).

:- pred other_app(list(T)::out, list(T)::in, list(T)::in) is semidet.
other_app(Prefix, Suffix, List) :-
    len(List, ListLength),
    len(Suffix, SuffixLength),
    PrefixLength = ListLength - SuffixLength,
    split(PrefixLength, List, Prefix, Suffix).

:- pred len(list(T)::in, int::out) is det.
len([], 0).
len([_ | Xs], N) :- len(Xs, N0), N = N0 + 1.

:- pred split(int::in, list(T)::in, list(T)::out, list(T)::out) is semidet.
split(N, List, Start, End) :-
    ( if N = 0 then
        Start = [], End = List
    else
        N > 0, List = [H | T], Start = [H | Start0],
        split(N - 1, T, Start0, End)
    ).

% Reversal both ways, with a mode written out.
:- pred rev(list(T), list(T)).
:- mode rev(in, out) is det.
:- mode rev(out, in) is det.
:- pragma promise_equivalent_clauses(rev/2).
rev(L::in, R::out) :- rev_acc(L, [], R).
rev(L::out, R::(ground >> ground)) :- rev_acc(R, [], L).

:- pred rev_acc(list(T)::in, list(T)::in, list(T)::out) is det.
rev_acc([], Acc, Acc).
rev_acc([X | Xs], Acc, R) :- rev_acc(Xs, [X | Acc], R).

% No clause is for half(out, in), which fails.
:- pred half(int, int).
:- mode half(in, out) is det.
:- mode half(out, in) is semidet.
:- pragma promise_equivalent_clauses(half/2).
half(X::in, Y::out) :- Y = X / 2.

% A function's default mode.
:- func twice(int) = int.
:- pragma promise_equivalent_clauses(twice/1).
twice(X::in) = (Y::out) :- Y = X * 2.

main(!IO) :-
    app([1, 2], [3], A), io.write(A, !IO), io.nl(!IO),
    ( if app(B, [3, 4], [1, 2, 3, 4]) then
        io.write(B, !IO)
    else
        io.write_string("no", !IO)
    ),
    io.nl(!IO),
    ( if app(C, [9], [1, 2, 3]) then
        io.write(C, !IO)
    else
        io.write_string("no", !IO)
    ),
    io.nl(!IO),
    ( if app([1], D, [1, 2, 3]) then io.write(D, !IO) else true ),
    io.nl(!IO),
    ( if app(E, F, [1, 2, 3]), len(E, 2) then
        io.write({E, F}, !IO)
    else
        io.write_string("no", !IO)
    ),
    io.nl(!IO),
    rev([1, 2, 3], G), io.write(G, !IO), io.nl(!IO),
    rev(H, [4, 5, 6]), io.write(H, !IO), io.nl(!IO),
    half(10, I), io.write_int(I, !IO), io.nl(!IO),
    ( if half(J, 5) then io.write_int(J, !IO) else io.write_string("no", !IO) ),
    io.nl(!IO),
    io.write_int(twice(21), !IO), io.nl(!IO).

% MERK-LABEL: merk.proc private @clauses_for_one_mode.app_3_0(
% MERK-NEXT: merk.call @clauses_for_one_mode.usual_app_3_0(
% MERK-LABEL: merk.proc private @clauses_for_one_mode.app_3_3(
% MERK-NEXT: merk.call @clauses_for_one_mode.other_app_3_0(
% MERK-LABEL: merk.proc private @clauses_for_one_mode.app_3_2(
% MERK-NEXT: merk.call @clauses_for_one_mode.usual_app_3_2(
% MERK-LABEL: merk.proc private @clauses_for_one_mode.app_3_1(
% MERK-NEXT: merk.nondet_call @clauses_for_one_mode.usual_app_3_1(
% MERK-LABEL: merk.proc private @clauses_for_one_mode.half_2_1(%arg0: i64) -> i64 attributes {determinism = #merk.determinism<semidet>} {
% MERK-NEXT: merk.fail
% MERK-NEXT: }

% OUTPUT:[1, 2, 3]
% OUTPUT-NEXT:[1, 2]
% OUTPUT-NEXT:no
% OUTPUT-NEXT:[2, 3]
% OUTPUT-NEXT:{[1, 2], [3]}
% OUTPUT-NEXT:[3, 2, 1]
% OUTPUT-NEXT:[6, 5, 4]
% OUTPUT-NEXT:5
% OUTPUT-NEXT:no
% OUTPUT-NEXT:42
