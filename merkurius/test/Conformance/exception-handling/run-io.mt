% MANUAL: Exception handling
% DESIGN: Sized floating point types
% The io try parameter threads the I/O state through GOAL. The manual's
% example, p_carefully, with a p that returns, one that throws a string, one
% that throws 42, and one that throws another value, which p_carefully
% rethrows. It is a template, run for float and float32.
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 %t.float.d/run_io_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/run_io_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:Calling p
% OUTPUT-FLOAT-NEXT:p returned: 0
% OUTPUT-FLOAT-NEXT:Calling p
% OUTPUT-FLOAT-NEXT:p threw a string: one
% OUTPUT-FLOAT-NEXT:Calling p
% OUTPUT-FLOAT-NEXT:p threw 42
% OUTPUT-FLOAT-NEXT:Calling p
% OUTPUT-FLOAT-NEXT:p threw something: 3.5
% OUTPUT-FLOAT-NEXT:main caught 3.5
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 %t.float32.d/run_io_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/run_io_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:Calling p
% OUTPUT-FLOAT32-NEXT:p returned: 0
% OUTPUT-FLOAT32-NEXT:Calling p
% OUTPUT-FLOAT32-NEXT:p threw a string: one
% OUTPUT-FLOAT32-NEXT:Calling p
% OUTPUT-FLOAT32-NEXT:p threw 42
% OUTPUT-FLOAT32-NEXT:Calling p
% OUTPUT-FLOAT32-NEXT:p threw something: 3.5f32
% OUTPUT-FLOAT32-NEXT:main caught 3.5
% OUTPUT-FLOAT32-NOT:{{.}}

:- module run_io.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module @TM@, exception, int, list, string.

:- pred p(int::in, int::out, io::di, io::uo) is det.
p(N, Output, !IO) :-
    ( if N = 1 then
        throw("one")
    else if N = 2 then
        throw(42)
    else if N = 3 then
        throw(3.5@S@)
    else
        Output = N * 10
    ).

:- pred p_carefully(int::in, io::di, io::uo) is cc_multi.
p_carefully(N, !IO) :-
    (try [io(!IO)] (
        io.write_string("Calling p\n", !IO),
        p(N, Output, !IO)
    )
    then
        io.write_string("p returned: ", !IO),
        io.write(Output, !IO),
        io.nl(!IO)
    catch S ->
        io.write_string("p threw a string: ", !IO),
        io.write_string(S, !IO),
        io.nl(!IO)
    catch 42 ->
        io.write_string("p threw 42\n", !IO)
    catch_any Other ->
        io.write_string("p threw something: ", !IO),
        io.write(Other, !IO),
        io.nl(!IO),
        % Rethrow the value.
        throw(Other)
    ).

main(!IO) :-
    p_carefully(0, !IO),
    p_carefully(1, !IO),
    p_carefully(2, !IO),
    ( try [io(!IO)] p_carefully(3, !IO)
    then
        true
    catch F ->
        io.format("main caught %.1f\n", [f(@TO_FLOAT@(F))], !IO)
    ).
