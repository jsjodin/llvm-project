% give0 and give1 return the same float, hidden by c0 and by c1, so their
% type_infos come from two typeclass_infos. At -O2 the optimizer makes one
% word of the two equal sums, which the bundles list with either type_info,
% and the statepoint pass must give it, where liveness did not list it, one
% that is available there. It took the last a bundle listed, defined after
% the call, and stopped with "a garbage collector root whose type_infos are
% not known". Found by the codegen oracle's generate-class-programs.py.
%
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 | FileCheck %s --match-full-lines --strict-whitespace
%
% CHECK:4.25
% CHECK-NEXT:'<'
% CHECK-NEXT:4.25
% CHECK-NEXT:6
% CHECK-EMPTY:

:- module merged_type_infos.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, list.

:- typeclass c0(T) where [
    pred bump0(int::in, T::in, T::out) is det
].
:- typeclass c1(T) <= c0(T) where [
    pred bump1(int::in, T::in, T::out) is det,
    func size1(T) = int
].
:- instance c0(float) where [
    (bump0(Delta, V, R) :- R = V + float(Delta))
].
:- instance c1(float) where [
    (bump1(Delta, V, R) :- R = V + float(Delta)),
    (size1(V) = truncate_to_int(V))
].
:- instance c0(list(T)) <= c0(T) where [
    (bump0(_, L, L))
].
:- instance c1(list(T)) <= c1(T) where [
    (bump1(_, L, L)),
    (size1(L) = list.foldl((func(X, S) = S + size1(X)), L, 0))
].

:- some [T] pred give0(int::in, T::out) is det => c0(T).
give0(_, 2.25 : float).

:- some [T] pred give1(int::in, T::out) is det => c1(T).
give1(_, 2.25 : float).

main(!IO) :-
    give0(1, A),
    bump0(2, A, AB), io.write(AB, !IO), io.nl(!IO),
    compare(R, A, AB), io.write(R, !IO), io.nl(!IO),
    give1(1, B),
    bump1(2, B, BB), io.write(BB, !IO), io.nl(!IO),
    io.write_int(size1([B, BB]), !IO), io.nl(!IO),
    io.nl(!IO).
