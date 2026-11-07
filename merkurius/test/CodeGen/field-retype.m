% Field updates that change the type of a type parameter only the field
% has (the register, "Type-changing field updates"): the new value goes in
% the cell, and the other fields stay as they are, as none of them has
% that parameter; in monomorphic and in polymorphic code, of a type of
% several constructors, in a list, and of one parameter of two. The output
% is mmc's for the same program.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module field_retype.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, list.

:- type box(T) ---> box(contents :: T, count :: int).
:- type u(T) ---> one(x :: T) ; other(y :: int, z :: T).
:- type nest(T) ---> nest(l :: list(T), n :: int).
:- type kv(K, V) ---> kv(k :: K, val :: V).

    % In polymorphic code: the result's type parameter is another.
:- func put(box(T), U) = box(U).
put(B, X) = B ^ contents := X.

main(!IO) :-
    B0 = box("s", 4),
    B1 = B0 ^ contents := 5,
    io.write(B1, !IO), io.nl(!IO),
    B2 = put(B1, 2.5),
    io.write(B2, !IO), io.nl(!IO),
    io.write(put(B2, [B0]), !IO), io.nl(!IO),
    ( if U = (one("a") ^ x := 'c') then io.write(U, !IO) else io.write_string("no", !IO) ),
    io.nl(!IO),
    ( if U2 = ((other(1, "b") : u(string)) ^ x := 'c') then io.write(U2, !IO) else io.write_string("no", !IO) ),
    io.nl(!IO),
    N = nest(["a"], 1) ^ l := [1.5, 2.5],
    io.write(N, !IO), io.nl(!IO),
    KV = kv(1, "one") ^ val := 1.0,
    io.write(KV, !IO), io.nl(!IO).

% OUTPUT:box(5, 4)
% OUTPUT-NEXT:box(2.5, 4)
% OUTPUT-NEXT:box([box("s", 4)], 4)
% OUTPUT-NEXT:one('c')
% OUTPUT-NEXT:no
% OUTPUT-NEXT:nest([1.5, 2.5], 1)
% OUTPUT-NEXT:kv(1, 1.0)
% OUTPUT-EMPTY:
