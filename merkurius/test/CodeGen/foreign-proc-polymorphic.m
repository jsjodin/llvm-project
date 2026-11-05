% A foreign_proc on a polymorphic procedure: a value of a type variable is an
% MR_Word, and TypeInfo_for_T is the type variable's type_info, named as the
% declaration names it. The manual describes neither; the output is mmc's for
% the same program.
%
% RUN: merkc %s -o %t && %t | diff %S/Inputs/foreign-proc-polymorphic.expected -
% RUN: env MERK_HEAP_CHUNK_BYTES=64 %t | diff %S/Inputs/foreign-proc-polymorphic.expected -
% RUN: merkc -O2 %s -o %t2 && %t2 | diff %S/Inputs/foreign-proc-polymorphic.expected -
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s

:- module foreign_proc_polymorphic.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

% CHECK-DAG: merk.proc private @foreign_proc_polymorphic.id_2_0(%arg0: !merk.type_info, %arg1: !merk.word) -> !merk.word
:- pred id(T::in, T::out) is det.
:- pragma foreign_proc("C", id(X::in, Y::out),
    [will_not_call_mercury, promise_pure], "Y = X;").

% Several type variables, in any order, and several outputs.
:- pred two(A::in, B::in, B::out, A::out) is det.
:- pragma foreign_proc("C", two(X::in, Y::in, Z::out, W::out),
    [will_not_call_mercury, promise_pure], "
    (void)TypeInfo_for_A;
    (void)TypeInfo_for_B;
    Z = Y;
    W = X;
").

% Whether two type variables' type_infos are the same word, for types with no
% arguments, which tells that they are the type_infos of those types.
:- pred same_type(A::in, B::in) is semidet.
:- pragma foreign_proc("C", same_type(_X::in, _Y::in),
    [will_not_call_mercury, promise_pure],
    "SUCCESS_INDICATOR = (TypeInfo_for_A == TypeInfo_for_B);").

% A type variable only within a type, as a function's result, by any name.
:- func f(list(Elem)) = list(Elem).
:- pragma foreign_proc("C", f(L::in) = (R::out),
    [will_not_call_mercury, promise_pure], "R = L; (void)TypeInfo_for_Elem;").

:- pred semi(T::in, T::out) is semidet.
:- pragma foreign_proc("C", semi(X::in, Y::out),
    [will_not_call_mercury, promise_pure], "Y = X; SUCCESS_INDICATOR = MR_TRUE;").

:- pred unused(int::in, T::in, int::out) is det.
:- pragma foreign_proc("C", unused(I::in, _X::in, O::out),
    [will_not_call_mercury, promise_pure], "O = I + 1;").

:- type box(T) ---> box(T).
:- pred b(box(T)::in, T::in, box(T)::out) is det.
:- pragma foreign_proc("C", b(X::in, _Y::in, Z::out),
    [will_not_call_mercury, promise_pure], "Z = X;").

% Two modes.
:- pred m(T, T).
:- mode m(in, out) is det.
:- mode m(out, in) is det.
:- pragma foreign_proc("C", m(X::in, Y::out),
    [will_not_call_mercury, promise_pure], "Y = X;").
:- pragma foreign_proc("C", m(X::out, Y::in),
    [will_not_call_mercury, promise_pure], "X = Y;").

% The manual's conversions of a word (C data passing conventions), for values
% that the code knows the types of.
:- pred float_of(T::in, float::out) is det.
:- pragma foreign_proc("C", float_of(X::in, F::out),
    [will_not_call_mercury, promise_pure], "F = MR_word_to_float(X);").
:- pred of_float(float::in, T::out) is det.
:- pragma foreign_proc("C", of_float(F::in, X::out),
    [will_not_call_mercury, promise_pure], "X = MR_float_to_word(F);").
:- pred int64_of(T::in, int64::out) is det.
:- pragma foreign_proc("C", int64_of(X::in, I::out),
    [will_not_call_mercury, promise_pure], "I = MR_word_to_int64(X);").
:- pred of_int64(int64::in, T::out) is det.
:- pragma foreign_proc("C", of_int64(I::in, X::out),
    [will_not_call_mercury, promise_pure], "X = MR_int64_to_word(I);").
:- pred uint64_of(T::in, uint64::out) is det.
:- pragma foreign_proc("C", uint64_of(X::in, I::out),
    [will_not_call_mercury, promise_pure], "I = MR_word_to_uint64(X);").
:- pred of_uint64(uint64::in, T::out) is det.
:- pragma foreign_proc("C", of_uint64(I::in, X::out),
    [will_not_call_mercury, promise_pure], "X = MR_uint64_to_word(I);").

% Called from polymorphic code, with its own type_info passed on.
:- pred via(T::in, io::di, io::uo) is det.
via(X, !IO) :-
    id(X, Y),
    id([Y], L),
    io.write(L, !IO),
    io.nl(!IO).

:- type dummy ---> dummy.

:- pred same_or_not(A::in, B::in, io::di, io::uo) is det.
same_or_not(X, Y, !IO) :-
    ( if same_type(X, Y) then
        io.write_string("same\n", !IO)
    else
        io.write_string("differ\n", !IO)
    ).

main(!IO) :-
    id(1.5, A), io.write_float(A, !IO), io.nl(!IO),
    id("s", S), io.write_string(S, !IO), io.nl(!IO),
    id('c', C), io.write_char(C, !IO), io.nl(!IO),
    id(7u8, U), io.write_uint8(U, !IO), io.nl(!IO),
    id(-3i16, I16), io.write_int16(I16, !IO), io.nl(!IO),
    id(dummy, D), io.write(D, !IO), io.nl(!IO),
    id({1, "t"}, T), io.write(T, !IO), io.nl(!IO),
    two(1, "x", Z, W), io.write_string(Z, !IO), io.write_int(W, !IO), io.nl(!IO),
    same_or_not(1, 2, !IO),
    same_or_not(1, "x", !IO),
    same_or_not("x", "y", !IO),
    same_or_not('c', 1.5, !IO),
    io.write(f([1, 2, 3]), !IO), io.nl(!IO),
    ( if semi(4, K) then io.write_int(K, !IO) else io.write_string("no", !IO) ), io.nl(!IO),
    unused(1, "z", O), io.write_int(O, !IO), io.nl(!IO),
    b(box(1), 2, box(Q)), io.write_int(Q, !IO), io.nl(!IO),
    m(5, M1), io.write_int(M1, !IO), io.nl(!IO),
    m(M2, "out"), io.write_string(M2, !IO), io.nl(!IO),
    float_of(2.5, F1), io.write_float(F1, !IO), io.nl(!IO),
    of_float(-0.25, F2), io.write_float(F2, !IO), io.nl(!IO),
    of_float(3.5, F3 : float), io.write(F3, !IO), io.nl(!IO),
    int64_of(-9i64, I1), io.write_int64(I1, !IO), io.nl(!IO),
    of_int64(-8i64, I2 : int64), io.write_int64(I2, !IO), io.nl(!IO),
    uint64_of(9u64, I3), io.write_uint64(I3, !IO), io.nl(!IO),
    of_uint64(8u64, I4 : uint64), io.write_uint64(I4, !IO), io.nl(!IO),
    via(1.5, !IO),
    via("v", !IO),
    via([box('x')], !IO).
