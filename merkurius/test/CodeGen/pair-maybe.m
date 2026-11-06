% The library's pair and maybe modules, compiled into libmerkstd: pairs and
% their projections, and maybe's higher-order maps and folds, with det and
% semidet closures and with the state of the world as an accumulator, and
% its other functions. The output is mmc's for the same program.
%
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module pair_maybe.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, pair.

:- pred add(int::in, int::in, int::out) is det.
add(X, A0, A) :- A = A0 + X.
:- pred add_io(int::in, io::di, io::uo) is det.
add_io(X, !IO) :- io.write_int(X, !IO), io.nl(!IO).
:- pred small(int::in, int::out) is semidet.
small(X, Y) :- X < 10, Y = X * 2.

main(!IO) :-
    Small = (pred(X::in, Y::out) is semidet :- small(X, Y)),
    Add = (pred(X::in, A0::in, A::out) is det :- add(X, A0, A)),
    AddIO = (pred(X::in, !.IO::di, !:IO::uo) is det :- add_io(X, !IO)),
    P = pair(1, "one"),
    io.write(P, !IO), io.nl(!IO),
    F0 = fst(P) : int, io.write(F0, !IO), io.write_string(" ", !IO),
    S0 = snd(P) : string, io.write(S0, !IO), io.nl(!IO),
    fst(P, F), snd(P, S), io.write({F, S}, !IO), io.nl(!IO),
    Ps = [1 - 'a', 2 - 'b'] : list(pair(int, character)),
    io.write(Ps, !IO), io.nl(!IO),
    io.write(map_maybe((func(X) = X + 1), yes(1)), !IO), io.nl(!IO),
    io.write(map_maybe((func(X) = X + 1), no : maybe(int)), !IO), io.nl(!IO),
    map_maybe((pred(X::in, Y::out) is det :- Y = X * 3), yes(2), M1),
    io.write(M1, !IO), io.nl(!IO),
    ( if map_maybe(Small, yes(20), M2) then io.write(M2, !IO) else io.write_string("fails", !IO) ),
    io.nl(!IO),
    ( if map_maybe(Small, yes(4), M3) then io.write(M3, !IO) else io.write_string("fails", !IO) ),
    io.nl(!IO),
    io.write(fold_maybe((func(X, A) = X + A), yes(2), 10), !IO), io.nl(!IO),
    io.write(fold_maybe((func(X, A) = X + A), no, 10), !IO), io.nl(!IO),
    fold_maybe(Add, yes(5), 100, A1), io.write(A1, !IO), io.nl(!IO),
    fold_maybe(AddIO, yes(42), !IO),
    fold_maybe(AddIO, no, !IO),
    fold2_maybe((pred(X::in, B0::in, B::out, C0::in, C::out) is det :-
        B = B0 + X, C = C0 * X), yes(3), 1, B2, 2, C2),
    io.write({B2, C2}, !IO), io.nl(!IO),
    fold3_maybe((pred(X::in, B0::in, B::out, C0::in, C::out, G0::in, G::out) is det :-
        B = B0 + X, C = C0 + X, G = G0 + X), yes(1), 1, B3, 2, C3, 3, G3),
    io.write({B3, C3, G3}, !IO), io.nl(!IO),
    map_fold_maybe((pred(X::in, Y::out, Q0::in, Q::out) is det :-
        Y = X * 10, Q = Q0 + 1), yes(4), M4, 0, Q4),
    io.write({M4, Q4}, !IO), io.nl(!IO),
    map_fold_maybe((pred(X::in, Y::out, !.IO::di, !:IO::uo) is det :-
        Y = X + 1, io.write_int(X, !IO), io.nl(!IO)), yes(8), M5, !IO),
    io.write(M5, !IO), io.nl(!IO),
    map_fold2_maybe((pred(X::in, Y::out, Q0::in, Q::out, R0::in, R::out) is det :-
        Y = X, Q = Q0 + X, R = R0 - X), no, M6 : maybe(int), 5, Q6, 5, R6),
    io.write({M6, Q6, R6}, !IO), io.nl(!IO),
    ( if maybe_is_yes(yes(5), V) then io.write(V, !IO) else io.write_string("no", !IO) ),
    io.nl(!IO),
    ( if maybe_is_yes(no : maybe(int), V2) then io.write(V2, !IO) else io.write_string("no", !IO) ),
    io.nl(!IO),
    io.write(pred_to_maybe((pred(X::out) is semidet :- X = 3)), !IO), io.nl(!IO),
    io.write(pred_to_maybe((pred(X::out) is semidet :- X = 3, X > 5)), !IO), io.nl(!IO),
    io.write(func_to_maybe(((func) = (Y::out) is semidet :- Y = 4)), !IO), io.nl(!IO),
    io.write(maybe_default(7, no), !IO), io.write_string(" ", !IO),
    io.write(maybe_default(7, yes(8)), !IO), io.nl(!IO),
    Err = error("bad") : maybe_error(int),
    io.write([ok(1), Err], !IO), io.nl(!IO),
    Es = error("first", ["second"]) : maybe_errors(int),
    io.write(Es, !IO), io.nl(!IO),
    io.write(ok : maybe_error, !IO), io.nl(!IO).

% OUTPUT:1 - "one"
% OUTPUT-NEXT:1 "one"
% OUTPUT-NEXT:{1, "one"}
% OUTPUT-NEXT:[1 - 'a', 2 - 'b']
% OUTPUT-NEXT:yes(2)
% OUTPUT-NEXT:no
% OUTPUT-NEXT:yes(6)
% OUTPUT-NEXT:fails
% OUTPUT-NEXT:yes(8)
% OUTPUT-NEXT:12
% OUTPUT-NEXT:10
% OUTPUT-NEXT:105
% OUTPUT-NEXT:42
% OUTPUT-NEXT:{4, 6}
% OUTPUT-NEXT:{2, 3, 4}
% OUTPUT-NEXT:{yes(40), 1}
% OUTPUT-NEXT:8
% OUTPUT-NEXT:yes(9)
% OUTPUT-NEXT:{no, 5, 5}
% OUTPUT-NEXT:5
% OUTPUT-NEXT:no
% OUTPUT-NEXT:yes(3)
% OUTPUT-NEXT:no
% OUTPUT-NEXT:yes(4)
% OUTPUT-NEXT:7 8
% OUTPUT-NEXT:[ok(1), error("bad")]
% OUTPUT-NEXT:error("first", ["second"])
% OUTPUT-NEXT:ok
