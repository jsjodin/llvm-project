% MANUAL: The universal type
% DESIGN: Sized floating point types
% A value of any type converts to univ and back with type_to_univ and
% univ_to_type; the conversion back checks the type, which makes a
% heterogeneous list possible. It is a template, run for float and float32.
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 %t.float.d/run_univ_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/run_univ_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:int 1
% OUTPUT-FLOAT-NEXT:string two
% OUTPUT-FLOAT-NEXT:something else
% OUTPUT-FLOAT-NEXT:something else
% OUTPUT-FLOAT-NEXT:something else
% OUTPUT-FLOAT-NEXT:43
% OUTPUT-FLOAT-NEXT:not a string
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 %t.float32.d/run_univ_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/run_univ_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:int 1
% OUTPUT-FLOAT32-NEXT:string two
% OUTPUT-FLOAT32-NEXT:something else
% OUTPUT-FLOAT32-NEXT:something else
% OUTPUT-FLOAT32-NEXT:something else
% OUTPUT-FLOAT32-NEXT:43
% OUTPUT-FLOAT32-NEXT:not a string
% OUTPUT-FLOAT32-NOT:{{.}}

:- module run_univ.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, list, string, univ.

main(!IO) :-
    Us = [univ(1), univ("two"), univ(3.0@S@), univ([4]), univ('5')],
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
