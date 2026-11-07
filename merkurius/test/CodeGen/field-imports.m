% Field accesses and updates of another module's types
% (Inputs/field_types.m), whose fields its interface declares: of a type
% of one constructor, and of a polymorphic one of two, where they can
% fail. The output is mmc's for the same program.
%
% RUN: merkc -O0 -I %S/Inputs %S/Inputs/field_types.m %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %S/Inputs/field_types.m %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module field_imports.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module field_types, int.

:- func item_or(entry(T), T) = T.
item_or(E, D) = ( if X = E ^ item then X else D ).

main(!IO) :-
    A0 = account("ann", 10),
    A = A0 ^ balance := A0 ^ balance + 5,
    io.write(A, !IO),
    io.nl(!IO),
    io.write_string(A ^ owner, !IO),
    io.nl(!IO),
    io.write_int(item_or(entry(1, 42), 0) + item_or(blank : entry(int), 7), !IO),
    io.nl(!IO),
    ( if E = ((blank : entry(int)) ^ key := 3) then
        io.write(E, !IO)
    else
        io.write_string("no key", !IO)
    ),
    io.nl(!IO).

% OUTPUT:account("ann", 15)
% OUTPUT-NEXT:ann
% OUTPUT-NEXT:49
% OUTPUT-NEXT:no key
% OUTPUT-EMPTY:
