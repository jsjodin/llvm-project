% MANUAL: Exception handling
% DESIGN: Sized floating point types
% If no catch's term unifies with the exception, the catch_any branch runs
% with the exception bound to its variable, which has type some [T] T. It is
% a template, run for float and float32.
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 %t.float.d/run_catch_any_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/run_catch_any_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:any: point(1, 2)
% OUTPUT-FLOAT-NEXT:any: 2.5
% OUTPUT-FLOAT-NEXT:the string
% OUTPUT-FLOAT-NEXT:any: "other"
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 %t.float32.d/run_catch_any_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/run_catch_any_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:any: point(1, 2)
% OUTPUT-FLOAT32-NEXT:any: 2.5f32
% OUTPUT-FLOAT32-NEXT:the string
% OUTPUT-FLOAT32-NEXT:any: "other"
% OUTPUT-FLOAT32-NOT:{{.}}

:- module run_catch_any.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module @TM@, exception, int, list, string.

:- type point
    --->    point(int, int).

:- pred show(T::in, io::di, io::uo) is cc_multi.
show(V, !IO) :-
    ( try [] throw(V)
    then
        io.write_string("then\n", !IO)
    catch "string" ->
        io.write_string("the string\n", !IO)
    catch_any E ->
        io.write_string("any: ", !IO),
        io.write(E, !IO),
        io.nl(!IO)
    ).

main(!IO) :-
    show(point(1, 2), !IO),
    show(2.5@S@, !IO),
    show("string", !IO),
    show("other", !IO).
