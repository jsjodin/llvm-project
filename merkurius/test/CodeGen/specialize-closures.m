% The specializer (docs/INPROGRESS.md, The specializer: known closures and
% dictionaries, step 1): at -O2, a call that passes a closure made by
% merk.closure to a procedure with a body calls a clone of it, which takes
% the closure's captured values, and the type_info of each, in place of the
% closure. In the clone, calling the closure is a call of its entry, cloned in
% turn, so a fold's loop calls its body directly. A closure put in a term
% stays one, and so does one deeper than three levels of closures captured by
% closures. A clone that comes out the same as its procedure is not kept. The
% output is mmc's for the same program.
%
% RUN: merkc -fc1 -O2 -emit-merk -triple x86_64-unknown-linux-gnu %s -o %t.merk
% RUN: FileCheck %s --check-prefix=FOLD < %t.merk
% RUN: FileCheck %s --check-prefix=EACH < %t.merk
% RUN: FileCheck %s --check-prefix=BOX < %t.merk
% RUN: FileCheck %s --check-prefix=COMPOSE < %t.merk
% RUN: FileCheck %s --check-prefix=NONDET < %t.merk
% RUN: FileCheck %s --check-prefix=CHAIN < %t.merk
% RUN: FileCheck %s --check-prefix=POLY < %t.merk
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=O0
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module specialize_closures.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions.

% A fold of the module's own, which calls its closure for each element.
:- func fold(func(T, A) = A, list(T), A) = A.
fold(_, [], A) = A.
fold(F, [X | Xs], A) = fold(F, Xs, F(X, A)).

% A lambda that captures two values.
:- func weighted(int, int, list(int)) = int.
weighted(K, B, L) = fold(func(X, A) = A + K * X + B, L, 0).

% A higher-order predicate that recurses through another one.
:- pred each(pred(T, io, io), list(T), io, io).
:- mode each(pred(in, di, uo) is det, in, di, uo) is det.
each(_, [], !IO).
each(P, [X | Xs], !IO) :- P(X, !IO), each(P, Xs, !IO).

% A closure put in a term, which stays a closure.
:- type box ---> box(func(int) = int).
:- func boxed(int) = box.
boxed(K) = box(func(X) = X + K).
:- func unbox(box, int) = int.
unbox(box(F), X) = F(X).

% A closure that captured a closure.
:- func compose(func(int) = int, func(int) = int) = (func(int) = int).
compose(F, G) = (func(X) = F(G(X))).
:- func apply_all(func(int) = int, list(int)) = list(int).
apply_all(_, []) = [].
apply_all(F, [X | Xs]) = [F(X) | apply_all(F, Xs)].

% A nondet closure, called by a predicate given a known one.
:- pred gen(pred(int)::in(pred(out) is nondet), int::out) is nondet.
gen(P, X) :- P(X), X mod 2 = 0.

:- pred first_even(int::in, int::out) is semidet.
first_even(Min, X) :-
    promise_equivalent_solutions [X] (
        gen((pred(Y::out) is nondet :- list.member(Y, [1, 3, 4, 6]), Y > Min),
            X)
    ).

% Closures that captured closures, deeper than the specializer follows: the
% innermost stays a closure call.
:- func chain(int) = list(int).
chain(K) = apply_all(F4, [1, 2]) :-
    F1 = (func(X) = X + K),
    F2 = (func(X) = F1(X) * 2),
    F3 = (func(X) = F2(X) + 3),
    F4 = (func(X) = F3(X) - 1).

% A closure that captured a known closure.
:- func twice_then(int, list(int)) = list(int).
twice_then(K, L) = apply_all(G, L) :-
    F = (func(X) = X * K),
    G = (func(X) = F(X) + 1).

% A polymorphic value that a lambda captures.
:- func pairs(T, list(int)) = list({T, int}).
pairs(V, L) = map_list(func(X) = {V, X}, L).
:- func map_list(func(A) = B, list(A)) = list(B).
map_list(_, []) = [].
map_list(F, [X | Xs]) = [F(X) | map_list(F, Xs)].

main(!IO) :-
    io.write_int(weighted(3, 1, [1, 2, 3, 4]), !IO), io.nl(!IO),
    Prefix = "item ",
    each((pred(X::in, !.IO::di, !:IO::uo) is det :-
        io.write_string(Prefix, !IO), io.write_int(X, !IO), io.nl(!IO)),
        [7, 8], !IO),
    io.write_int(unbox(boxed(5), 10), !IO), io.nl(!IO),
    io.write(apply_all(compose(func(X) = X * 2, func(Y) = Y + 1), [1, 2, 3]),
        !IO), io.nl(!IO),
    solutions(gen((pred(X::out) is nondet :- list.member(X, [1, 2, 3, 4]))),
        Evens),
    io.write(Evens, !IO), io.nl(!IO),
    io.write(pairs("a", [1, 2]), !IO), io.nl(!IO),
    ( if first_even(3, E) then
        io.write_int(E, !IO)
    else
        io.write_string("none", !IO)
    ),
    io.nl(!IO),
    io.write(chain(10), !IO), io.nl(!IO),
    io.write(twice_then(3, [1, 2]), !IO), io.nl(!IO).

% The lambda's cell is gone: the fold's clone takes K and B, each with its
% type_info, and calls the lambda's entry clone, and itself.
% FOLD-LABEL: merk.proc private @specialize_closures.weighted_f3_0(
% FOLD-NOT: merk.closure
% FOLD: merk.call @specialize_closures.fold_f3_0$spec1(%{{.*}}) : (!merk.type_info, !merk.type_info, !merk.word, !merk.type_info, !merk.word, !merk.type_info, !merk.term<"list.list(T0)">, !merk.word) -> !merk.word
% FOLD-LABEL: merk.proc private @specialize_closures.fold_f3_0$spec1(
% FOLD-NOT: merk.call_closure
% FOLD: merk.call @specialize_closures.weighted_f3_0$lambda{{[0-9]+}}$closure{{[0-9]+}}$spec1(
% FOLD-NOT: merk.call_closure
% FOLD: merk.call @specialize_closures.fold_f3_0$spec1(
% FOLD-LABEL: merk.proc private @specialize_closures.weighted_f3_0$lambda{{[0-9]+}}$closure{{[0-9]+}}$spec1(
% FOLD-NOT: merk.closure_word
% FOLD: merk.succeed

% A higher-order predicate with io: the lambda's captured string.
% EACH-LABEL: merk.proc private @specialize_closures.each_4_0$spec1(%arg0: !merk.type_info, %arg1: !merk.word, %arg2: !merk.type_info, %arg3: !merk.term<"list.list(T0)">)
% EACH: merk.call @specialize_closures.main_2_0$lambda{{[0-9]+}}$closure{{[0-9]+}}$spec1(%arg1, %arg2,
% EACH: merk.call @specialize_closures.each_4_0$spec1(%arg0, %arg1, %arg2,

% BOX-LABEL: merk.proc private @specialize_closures.boxed_f1_0(
% BOX: merk.closure @specialize_closures.boxed_f1_0$lambda
% BOX-LABEL: merk.proc private @specialize_closures.unbox_f2_0(
% BOX: merk.call_closure

% COMPOSE-NOT: @specialize_closures.compose_f2_0$spec

% NONDET-LABEL: merk.proc private @specialize_closures.gen_2_0$spec1(
% NONDET-NOT: merk.nondet_call_closure
% NONDET: merk.nondet_call @specialize_closures.first_even_2_0$lambda{{[0-9]+}}$closure{{[0-9]+}}$spec1(

% Only F1, three levels down, is still made, and called as a closure by F2.
% CHAIN-LABEL: merk.proc private @specialize_closures.chain_f1_0(
% CHAIN-COUNT-1: merk.closure @
% CHAIN-NOT: merk.closure @
% CHAIN: merk.call @specialize_closures.apply_all_f2_0$spec{{[0-9]+}}(
% CHAIN: merk.proc private @specialize_closures.chain_f1_0$lambda{{[0-9]+}}$closure1$spec1(%arg0: !merk.word, %arg1: !merk.type_info, %arg2: !merk.word)
% CHAIN-NEXT: merk.from_word %arg0 : !merk.closure

% A captured value of a type variable: its type_info is the captured
% type_info's.
% POLY-LABEL: merk.proc private @specialize_closures.pairs_f2_0(
% POLY-NOT: merk.closure @
% POLY: merk.call @specialize_closures.map_list_f2_0$spec1(

% O0-NOT: $spec

% OUTPUT:34
% OUTPUT-NEXT:item 7
% OUTPUT-NEXT:item 8
% OUTPUT-NEXT:15
% OUTPUT-NEXT:[4, 6, 8]
% OUTPUT-NEXT:[2, 4]
% OUTPUT-NEXT:[{"a", 1}, {"a", 2}]
% OUTPUT-NEXT:4
% OUTPUT-NEXT:[24, 26]
% OUTPUT-NEXT:[4, 7]
