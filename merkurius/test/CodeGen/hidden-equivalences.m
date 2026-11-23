% An abstract type that its module defines as an equivalence is, in every
% module, the type it is once types are checked (reference manual,
% "Abstract types"): its values pass as that type's do, a float's in a
% float register, and its type_info is that type's, so that univ_to_type
% to that type succeeds, as with mmc, whose .int files carry the
% equivalences. A module that imports another whose interface uses the
% types sees through them too. The OUTPUT lines are mmc's.
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/hidden_equivalences.m %S/Inputs/hidden_wrapper.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/hidden_equivalences.m %S/Inputs/hidden_wrapper.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module hidden_equivalences_client.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module hidden_equivalences, hidden_wrapper, list, univ.

main(!IO) :-
    W = add(weight(1.25), weight(2.5)),
    io.write_line(to_float(W), !IO),
    io.write_line(W, !IO),
    io.write_line(univ(W), !IO),
    ( if univ_to_type(univ(W), F : float) then
        io.write_line(F, !IO)
    else
        io.write_string("not a float\n", !IO)
    ),
    S = next(next(small(254))),
    io.write_line(S, !IO),
    B = put("b", bag("a")),
    io.write_line(size(B), !IO),
    io.write_line(B, !IO),
    io.write_line(describe(heavy(W)), !IO),
    Ws = [weight(1.0), W, weight(0.5)],
    list.sort(Ws, Sorted),
    io.write_line(Sorted, !IO),
    P = parcel(2.0, "box"),
    io.write_line(P, !IO),
    io.write_line(to_float(total(P, P)), !IO),
    ( if W = add(weight(2.5), weight(1.25)) then
        io.write_string("equal\n", !IO)
    else
        io.write_string("unequal\n", !IO)
    ).

% OUTPUT:3.75
% OUTPUT-NEXT:3.75
% OUTPUT-NEXT:univ_cons(3.75)
% OUTPUT-NEXT:3.75
% OUTPUT-NEXT:0u8
% OUTPUT-NEXT:2
% OUTPUT-NEXT:[{"b", 2}, {"a", 1}]
% OUTPUT-NEXT:"37.5"
% OUTPUT-NEXT:[0.5, 1.0, 3.75]
% OUTPUT-NEXT:parcel(2.0, [{"box", 1}])
% OUTPUT-NEXT:4.0
% OUTPUT-NEXT:equal
% OUTPUT-NOT:{{.}}
