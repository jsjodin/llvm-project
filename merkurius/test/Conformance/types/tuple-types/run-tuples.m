% MANUAL: Tuple types
% The tuple types {}, {T}, {T1, T2} and so on: tuples are built, taken
% apart, nested, unified and compared argument by argument, as a
% discriminated union with one constructor {}/N would be.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{{[{][{]}}}, { 1 }, {2, "two"}, {{[{][{]}} 1 }, {2, "two"}, 'c'}}
% OUTPUT-NEXT:3
% OUTPUT-NEXT:"two"
% OUTPUT-NEXT:equal
% OUTPUT-NEXT:{(>), (<)}
% OUTPUT-NOT:{{.}}

:- module run_tuples.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, string.

main(!IO) :-
    T0 = {},
    T1 = {1},
    T2 = {2, "two"},
    T3 = {T1, T2, 'c'},
    io.write_line({T0, T1, T2, T3}, !IO),
    T3 = {{One}, {Two, Name}, _},
    io.write_line(One + Two, !IO),
    io.write_line(Name, !IO),
    ( if {1, "a"} = {1, "a"} then
        io.write_string("equal\n", !IO)
    else
        io.write_string("unequal\n", !IO)
    ),
    compare(R1, {1, "b"}, {1, "a"}),
    compare(R2, {1, "z"}, {2, "a"}),
    io.write_line({R1, R2}, !IO).
