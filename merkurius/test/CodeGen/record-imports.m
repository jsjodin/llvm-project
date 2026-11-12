% Arrays of flat records across modules (Inputs/record_tiles.m). A record
% type that the other module exports concretely is inline in this module's
% arrays as in that module's, with the same struct, since both decide from
% the type's merk.type_def by one rule (lib/CodeGen/Records.h): arrays it
% makes, this module updates and reads, and the other way round. A type it
% exports abstractly, and one with a field of such a type, are not records
% in either module: this one has no merk.type_def for them, and that one's
% says they are abstract. The output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-llvm -I %S/Inputs %s -o - | FileCheck %s --check-prefix=IMPORTS
% RUN: merkc -fc1 -emit-llvm %S/Inputs/record_tiles.m -o - | FileCheck %s --check-prefix=TILES
% RUN: merkc -O0 -I %S/Inputs %S/Inputs/record_tiles.m %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %S/Inputs/record_tiles.m %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t0 2>/dev/null | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% IMPORTS: getelementptr { i64, i64, i32 }, ptr addrspace(1) %{{.*}}
% TILES: getelementptr { i64, i64, i32 }, ptr addrspace(1) %{{.*}}

% OUTPUT:array([tile(0, 0, red), tile(1, 101, blue), tile(2, 4, red), tile(30, 7, blue), tile(4, 16, red)])
% OUTPUT-NEXT:128
% OUTPUT-NEXT:7
% OUTPUT-NEXT:6
% OUTPUT-NEXT:12
% OUTPUT-NEXT:18
% OUTPUT-NEXT:array([shaded(0, light), shaded(9, dark), shaded(2, light)])
% OUTPUT-NEXT:"same"
% OUTPUT-NOT:{{.}}

:- module record_imports.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, record_tiles.

main(!IO) :-
    A0 = tiles(5),
    bump(1, A0, A1),
    A2 = A1 ^ elem(3) := tile(30, 7, blue),
    io.write_line(A2, !IO),
    io.write_line(live_sum(A2), !IO),
    io.write_line(A2 ^ elem(3) ^ live, !IO),
    B = array.from_list([tile(1, 2, red), tile(3, 4, blue)]),
    io.write_line(live_sum(B), !IO),
    io.write_line(opaque_sum(opaques(4)), !IO),
    O = opaques(4),
    O1 = O ^ elem(0) := O ^ elem(3),
    io.write_line(opaque_sum(O1), !IO),
    S = shadeds(3),
    S1 = S ^ elem(1) := shaded(9, dark_shade),
    io.write_line(S1, !IO),
    ( if S1 ^ elem(1) = shaded(9, dark_shade) then
        io.write_line("same", !IO)
    else
        io.write_line("differ", !IO)
    ).
