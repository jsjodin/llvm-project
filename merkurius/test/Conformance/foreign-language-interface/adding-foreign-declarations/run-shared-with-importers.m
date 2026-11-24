% MANUAL: Adding foreign declarations
% A foreign_decl that is not local is seen by the foreign code of the
% modules that import its module.
% MODULES: Inputs/foreign_decl_shared.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/foreign_decl_shared.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/foreign_decl_shared.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:9
% OUTPUT-NEXT:10
% OUTPUT-NOT:{{.}}

:- module run_shared_with_importers.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module foreign_decl_shared.
:- func shared = int.
:- pragma foreign_proc("C", shared = (X::out),
    [promise_pure, will_not_call_mercury], "X = SHARED_ONE + SHARED_TWO;").

main(!IO) :-
    io.write_int(shared, !IO), io.nl(!IO),
    io.write_int(twice(5), !IO), io.nl(!IO).
