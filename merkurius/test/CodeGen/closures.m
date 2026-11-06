% Closures (design document, Lowering, "Closures"): a closure is a cell of its
% entry's code, the number of values it captured and those values. A lambda
% expression is a procedure that takes the values it captures, its body's
% type_infos and typeclass_infos among them, and then its inputs, and gives a
% word for each output; a closure of a named procedure captures what the call
% would pass and the curried arguments. A closure of a procedure of this
% module calls its word copy, and one of a procedure built into code
% generation has a body of its own. A higher-order call passes a word for each
% argument. The output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_HEAP_CHUNK_BYTES=64 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module closures.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, float, list.

:- typeclass shape(T) where [ func area(T) = float, func name(T) = string ].
:- type sq ---> sq(float).
:- instance shape(sq) where [ area(sq(S)) = S * S, name(_) = "square" ].
:- instance shape(int) where [ area(N) = float(N), name(_) = "int" ].

:- pred my_map(pred(A, B)::in(pred(in, out) is det), list(A)::in, list(B)::out) is det.
my_map(_, [], []).
my_map(P, [X | Xs], [Y | Ys]) :- P(X, Y), my_map(P, Xs, Ys).

:- func my_mapf(func(A) = B, list(A)) = list(B).
my_mapf(_, []) = [].
my_mapf(F, [X | Xs]) = [F(X) | my_mapf(F, Xs)].

:- pred my_foldl(pred(A, B, B)::in(pred(in, in, out) is det), list(A)::in, B::in, B::out) is det.
my_foldl(_, [], !Acc).
my_foldl(P, [X | Xs], !Acc) :- P(X, !Acc), my_foldl(P, Xs, !Acc).

:- func my_foldf(func(A, B) = B, list(A), B) = B.
my_foldf(_, [], Acc) = Acc.
my_foldf(F, [X | Xs], Acc) = my_foldf(F, Xs, F(X, Acc)).

:- pred my_filter(pred(A)::in(pred(in) is semidet), list(A)::in, list(A)::out) is det.
my_filter(_, [], []).
my_filter(P, [X | Xs], Ys) :-
    my_filter(P, Xs, Ys0),
    ( if P(X) then Ys = [X | Ys0] else Ys = Ys0 ).

:- pred my_foldl_io(pred(A, io, io)::in(pred(in, di, uo) is det), list(A)::in, io::di, io::uo) is det.
my_foldl_io(_, [], !IO).
my_foldl_io(P, [X | Xs], !IO) :- P(X, !IO), my_foldl_io(P, Xs, !IO).

% A lambda in a polymorphic predicate, which captures T's type_info.
:- pred count_equal(T::in, list(T)::in, int::out) is det.
count_equal(X, L, N) :-
    my_filter((pred(Y::in) is semidet :- Y = X), L, Same),
    N = len(Same).

:- func len(list(T)) = int.
len([]) = 0.
len([_ | T]) = 1 + len(T).

% A lambda that calls a method, capturing the typeclass_info.
:- func total_area(list(T)) = float <= shape(T).
total_area(L) = my_foldf(func(S, A) = A + area(S), L, 0.0).

% A closure of a constrained predicate, which captures the typeclass_info.
:- pred describe(string::in, T::in, {string, string, float}::out) is det <= shape(T).
describe(Prefix, S, {Prefix, name(S), area(S)}).

% A string without the string module, which is not compiled yet.
:- func show(int) = string.
show(N) = ( if N = 42 then "forty-two" else "other" ).

:- func size(string) = int.
size(S) = ( if S = "a" then 1 else if S = "bb" then 2 else 3 ).

:- pred add(int::in, int::in, int::out) is det.
add(X, Y, X + Y).

:- func scale(float, float) = float.
scale(K, X) = K * X.

:- pred between(int::in, int::in, int::in) is semidet.
between(Lo, Hi, X) :- Lo =< X, X =< Hi.

:- pred write_pair(string::in, int::in, io::di, io::uo) is det.
write_pair(Name, N, !IO) :- io.write_string(Name, !IO), io.write_string("=", !IO), io.write_int(N, !IO).

:- func compose(func(B) = C, func(A) = B) = (func(A) = C).
compose(F, G) = (func(X) = F(G(X))).

:- func twice(func(T) = T) = (func(T) = T).
twice(F) = compose(F, F).

:- pred try(pred(int, int)::in(pred(in, out) is semidet), int::in, int::out) is det.
try(P, X, Y) :- ( if P(X, Z) then Y = Z else Y = -1 ).

main(!IO) :-
    my_map(add(10), [1, 2, 3], L1), io.write(L1, !IO), io.nl(!IO),
    K = 3, S = "s",
    my_map((pred(X::in, Y::out) is det :- Y = {S, X * K}), [4, 5], L2),
    io.write(L2, !IO), io.nl(!IO),
    L3 = my_mapf(scale(2.5), [1.0, 2.0]), io.write(L3, !IO), io.nl(!IO),
    L4 = my_mapf(func(X) = X * 1.5 + float(K), [2.0]), io.write(L4, !IO), io.nl(!IO),
    my_foldl(add, [1, 2, 3, 4], 0, Sum), io.write_int(Sum, !IO), io.nl(!IO),
    Product = my_foldf(int.times, [1, 2, 3, 4], 1), io.write_int(Product, !IO), io.nl(!IO),
    Plus = my_foldf(int.plus, [5, 6], 0), io.write_int(Plus, !IO), io.nl(!IO),
    my_filter(between(2, 4), [1, 2, 3, 4, 5], L5), io.write(L5, !IO), io.nl(!IO),
    my_filter((pred(W::in) is semidet :- size(W) > 1), ["a", "bb", "ccc"], L6),
    io.write(L6, !IO), io.nl(!IO),
    count_equal("x", ["x", "y", "x"], C1), io.write_int(C1, !IO), io.nl(!IO),
    count_equal(2.0, [1.0, 2.0, 2.0, 2.0], C2), io.write_int(C2, !IO), io.nl(!IO),
    io.write_float(total_area([sq(1.0), sq(2.0)]), !IO), io.nl(!IO),
    io.write_float(total_area([1, 2, 3]), !IO), io.nl(!IO),
    my_map(describe("> "), [sq(3.0)], L7), io.write(L7, !IO), io.nl(!IO),
    my_map(describe(": "), [7], L8), io.write(L8, !IO), io.nl(!IO),
    io.write_list([1, 2, 3], ", ", io.write_int, !IO), io.nl(!IO),
    io.write_list([1, 2], "; ", write_pair("k"), !IO), io.nl(!IO),
    io.write_list([] : list(int), ", ", io.write_int, !IO), io.nl(!IO),
    my_foldl_io((pred(N::in, !.IO::di, !:IO::uo) is det :-
        io.write_int(N * K, !IO), io.write_string(" ", !IO)), [1, 2], !IO),
    io.nl(!IO),
    Inc = (func(X) = X + 1),
    F2 = twice(Inc), io.write_int(F2(5), !IO), io.nl(!IO),
    F3 = compose(show, Inc), io.write_string(apply(F3, 41), !IO), io.nl(!IO),
    Fs = [Inc, twice(Inc), (func(X) = X * K)],
    L9 = my_mapf((func(F) = F(10)), Fs), io.write(L9, !IO), io.nl(!IO),
    try((pred(A::in, B::out) is semidet :- A > 0, B = A * 2), 4, T1),
    try((pred(A::in, B::out) is semidet :- A > 0, B = A * 2), -4, T2),
    io.write_int(T1, !IO), io.write_string(" ", !IO), io.write_int(T2, !IO), io.nl(!IO).

% OUTPUT:[11, 12, 13]
% OUTPUT-NEXT:[{"s", 12}, {"s", 15}]
% OUTPUT-NEXT:[2.5, 5.0]
% OUTPUT-NEXT:[6.0]
% OUTPUT-NEXT:10
% OUTPUT-NEXT:24
% OUTPUT-NEXT:11
% OUTPUT-NEXT:[2, 3, 4]
% OUTPUT-NEXT:["bb", "ccc"]
% OUTPUT-NEXT:2
% OUTPUT-NEXT:3
% OUTPUT-NEXT:5.0
% OUTPUT-NEXT:6.0
% OUTPUT-NEXT:[{"> ", "square", 9.0}]
% OUTPUT-NEXT:[{": ", "int", 7.0}]
% OUTPUT-NEXT:1, 2, 3
% OUTPUT-NEXT:k=1; k=2
% OUTPUT-EMPTY:
% OUTPUT-NEXT:3 6 
% OUTPUT-NEXT:7
% OUTPUT-NEXT:forty-two
% OUTPUT-NEXT:[11, 12, 30]
% OUTPUT-NEXT:8 -1

% A closure of add/3 with one argument, through its word copy's entry.
% MERK-DAG: merk.closure @closures.add_3_0$words$closure1(%{{[0-9]+}})
% MERK-DAG: merk.proc private @closures.add_3_0$words(%arg0: i64, %arg1: i64) -> !merk.word
% MERK-DAG: merk.proc private @closures.add_3_0$words$closure1(%arg0: !merk.closure, %arg1: !merk.word) -> !merk.word
% MERK-DAG: merk.closure_word %arg0, 0
% A lambda expression's procedure takes the two values it captures first.
% MERK-DAG: merk.proc private @closures.main_2_0$lambda0(%arg0: i64, %arg1: !merk.string, %arg2: i64) -> !merk.word
% Built in, with a body of its own.
% MERK-DAG: merk.closure @merk.builtin.int.plus_f2_0$closure0
% MERK-DAG: merk.proc private @merk.builtin.int.plus_f2_0$closure0(%arg0: !merk.closure, %arg1: !merk.word, %arg2: !merk.word) -> !merk.word
% The type_info of T, and the typeclass_info of shape(sq), captured first.
% MERK-DAG: merk.closure @closures.count_equal_3_0$lambda{{[0-9]+}}$closure2(%0, %arg1)
% MERK-DAG: %[[D:[0-9]+]] = merk.to_word %{{[0-9]+}} : !merk.typeclass_info
% MERK-DAG: merk.closure @closures.describe_3_0$words$closure2(%[[D]], %{{[0-9]+}})
% MERK-DAG: merk.call_closure %arg2(%{{.*}}) <det> : (!merk.word) -> !merk.word
% A list of closures has the type_info of their type, whose descriptor stops
% a unification or comparison of them.
% MERK-DAG: merk.type_info "func/2"(%{{[0-9]+}}, %{{[0-9]+}})
% MERK-DAG: merk.type_ctor "func/2" module "" name "func" kind "closure" linkage "linkonce" arity 2 unify @"unify_entry.func/2" compare @"compare_entry.func/2"
