% A root whose type's fields name a type that nothing in this module does
% (Inputs/root_runs.m): r_steps deconstructs a run, whose other fields are
% arrays of root_runs.sample, and the run stays live across the calls that
% follow, so the root table must describe array(root_runs.sample) as well,
% from the descriptor that root_runs exports (MLIRGenTypeInfo.cpp,
% EnsureRootDescriptors). Lowering stopped at it as a root the runtime
% could not describe. The output is mmc's for the same program, and holds
% when the collector runs at every allocation.
%
% RUN: merkc -O0 -I %S/Inputs %S/Inputs/root_runs.m %s -o %t0 && %t0 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %S/Inputs/root_runs.m %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t0 2>/dev/null | FileCheck %s --match-full-lines --strict-whitespace

% CHECK:[1, 2, 3]
% CHECK-NEXT:3
% CHECK-NEXT:[1, 2, 3]
% CHECK-NEXT:3
% CHECK-NOT:{{.}}

:- module root_field_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, root_runs.

main(!IO) :-
    R = make_run(3),
    show(R, !IO),
    show(R, !IO).

:- pred show(run::in, io::di, io::uo) is det.

show(R, !IO) :-
    Steps = R ^ r_steps,
    io.write_line(Steps, !IO),
    io.write_line(list.length(Steps) : int, !IO).
