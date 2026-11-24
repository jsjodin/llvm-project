% MANUAL: Semantics of type quantifiers
% A caller treats an existentially typed value as opaque: it unifies it
% with another of the same type, puts it in a polymorphic structure, and
% passes it to universally quantified predicates such as io.write. Mode
% analysis reorders a conjunction so that the call that binds the type
% comes first.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:["hidden", "hidden"]
% OUTPUT-NEXT:same
% OUTPUT-NOT:{{.}}

:- module run_opaque.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- some [T] pred e_foo(T).
:- mode e_foo(out) is det.
e_foo("hidden").

:- pred same(T::in, T::in) is semidet.
same(X, X).

main(!IO) :-
    Y = X,
    e_foo(X),
    L = [X, Y],
    io.write_line(L, !IO),
    ( if same(X, Y) then
        io.write_string("same\n", !IO)
    else
        io.write_string("different\n", !IO)
    ).
