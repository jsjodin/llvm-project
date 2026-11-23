% MANUAL: The universal type
% A value of any type converts to univ and back with type_to_univ and
% univ_to_type; the conversion back checks the type, which makes a
% heterogeneous list possible.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:int 1
% OUTPUT-NEXT:string two
% OUTPUT-NEXT:something else
% OUTPUT-NEXT:something else
% OUTPUT-NEXT:something else
% OUTPUT-NEXT:43
% OUTPUT-NEXT:not a string
% OUTPUT-NOT:{{.}}

:- module run_univ.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string, univ.

main(!IO) :-
    Us = [univ(1), univ("two"), univ(3.0), univ([4]), univ('5')],
    list.foldl(describe, Us, !IO),
    type_to_univ(42, U),
    ( if univ_to_type(U, N) then
        io.write_line(N + 1, !IO)
    else
        io.write_string("not an int\n", !IO)
    ),
    ( if univ_to_type(U, S : string) then
        io.write_line(S, !IO)
    else
        io.write_string("not a string\n", !IO)
    ).

:- pred describe(univ::in, io::di, io::uo) is det.

describe(U, !IO) :-
    ( if univ_to_type(U, I : int) then
        io.write_string("int " ++ string.int_to_string(I) ++ "\n", !IO)
    else if univ_to_type(U, S : string) then
        io.write_string("string " ++ S ++ "\n", !IO)
    else
        io.write_string("something else\n", !IO)
    ).
