% An argument of mode free >> free carries nothing either way, so it is not
% passed, as one of a type with no values is not, and as mmc passes it: not
% in a procedure's signature, its calls, its foreign_proc's C function or a
% method's dispatch. A closure's entry, whose arguments are all words,
% takes a word for it and does not pass it on. The output is mmc's.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module free_free_arguments.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pred q(int::(free >> free)) is det.
q(_).

:- pred r(int::in, int::(free >> free), int::out) is det.
r(A, _, B) :-
    B = A + 1.

:- pred c(int::in, int::(free >> free), int::out) is det.
:- pragma foreign_proc("C",
    c(N::in, _U::(free >> free), M::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    M = N * 3;
").

:- typeclass shape(T) where [
    pred area(T::in, int::(free >> free), int::out) is det
].
:- type square ---> square(int).
:- instance shape(square) where [
    (area(square(S), _, A) :- A = S * S)
].

:- pred apply_area(T::in, int::out) is det <= shape(T).
apply_area(Shape, A) :-
    area(Shape, _, A).

:- pred call_free(pred(int, int, int)::in(pred(in, free >> free, out) is det),
    int::in, int::out) is det.
call_free(P, A, B) :-
    P(A, _, B).

main(!IO) :-
    q(X),
    X = 3,
    io.write_int(X, !IO),
    io.nl(!IO),
    r(10, Y, Z),
    Y = 4,
    io.write_int(Y + Z, !IO),
    io.nl(!IO),
    c(5, _, C),
    io.write_int(C, !IO),
    io.nl(!IO),
    call_free(r, 20, B1),
    io.write_int(B1, !IO),
    io.nl(!IO),
    call_free((pred(I::in, _::(free >> free), O::out) is det :- O = I * 2),
        21, B2),
    io.write_int(B2, !IO),
    io.nl(!IO),
    apply_area(square(7), A),
    io.write_int(A, !IO),
    io.nl(!IO).

% MERK: MR_Integer merk_foreign_free__free__arguments_Dc__3__0(MR_Integer N) {
% MERK-LABEL: merk.proc private @free_free_arguments.c_3_0(%arg0: i64) -> i64
% MERK-LABEL: merk.proc @free_free_arguments.main_2_0()
% MERK: merk.call @free_free_arguments.q_1_0() : () -> ()
% MERK: merk.call @free_free_arguments.r_3_0(%{{.*}}) : (i64) -> i64
% MERK: merk.call @free_free_arguments.c_3_0(%{{.*}}) : (i64) -> i64
% MERK-LABEL: merk.proc private @free_free_arguments.q_1_0() attributes
% MERK-LABEL: merk.proc private @free_free_arguments.r_3_0(%arg0: i64) -> i64
% MERK-LABEL: merk.proc private @free_free_arguments.r_3_0$words$closure0(%arg0: !merk.closure, %arg1: !merk.word, %arg2: !merk.word) -> !merk.word
% MERK: merk.call @free_free_arguments.r_3_0$words(%{{.*}}) : (i64) -> !merk.word
% MERK-LABEL: merk.proc private @free_free_arguments.call_free_3_0(
% MERK: merk.call_closure %arg0(%{{.*}}, %{{.*}}) <det> : (!merk.word, !merk.word) -> !merk.word
% MERK-LABEL: merk.proc private @free_free_arguments.main_2_0$lambda0(%arg0: i64) -> !merk.word
% MERK-LABEL: merk.proc private @free_free_arguments.apply_area_2_0(
% MERK: merk.call_method %arg0[0] (%arg1) <det> : (!merk.word) -> i64

% OUTPUT:3
% OUTPUT-NEXT:15
% OUTPUT-NEXT:15
% OUTPUT-NEXT:21
% OUTPUT-NEXT:42
% OUTPUT-NEXT:49
% OUTPUT-EMPTY:
