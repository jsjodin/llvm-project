% MANUAL: Type class constraints on instance declarations
% The manual's portrayable class: the instance for list(T) is constrained
% by portrayable(T), and portrays each element with its type's method.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1 2 3{{ }}
% OUTPUT-NEXT:a b  c{{  }}
% OUTPUT-NOT:{{.}}

:- module run_portrayable.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, list.

:- typeclass portrayable(T) where [
    pred portray(T::in, io.state::di, io.state::uo) is det
].

:- instance portrayable(int) where [
    pred(portray/3) is io.write_int
].

:- instance portrayable(char) where [
    pred(portray/3) is io.write_char
].

:- instance portrayable(list(T)) <= portrayable(T) where [
    pred(portray/3) is portray_list
].

:- pred portray_list(list(T), io.state, io.state) <= portrayable(T).
:- mode portray_list(in, di, uo) is det.

portray_list([], !IO).
portray_list([X | Xs], !IO) :-
    portray(X, !IO),
    io.write_char(' ', !IO),
    portray_list(Xs, !IO).

main(!IO) :-
    portray([1, 2, 3], !IO),
    io.nl(!IO),
    portray([['a', 'b'], ['c']], !IO),
    io.nl(!IO).
