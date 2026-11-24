% The procedures of initialise and finalise directives (reference manual,
% "Module initialisation" and "Module finalisation") are merk.initialise
% ops, in their order. Each lowers to a C function that calls the
% procedure, which one C constructor registers with the runtime, which
% calls the initialisers before main and the finalisers after.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=LLVM
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module init_directives.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- initialise first/2.
:- finalise last/0.
:- initialize second/0.

:- pred first(io::di, io::uo) is det.
first(!IO) :-
    io.write_string("first\n", !IO).

:- impure pred second is det.
second :-
    impure impure_true,
    trace [io(!IO)] (
        io.write_string("second\n", !IO)
    ).

:- impure pred last is cc_multi.
:- pragma no_determinism_warning(last/0).
last :-
    impure impure_true,
    trace [io(!IO)] (
        io.write_string("last\n", !IO)
    ).

main(!IO) :-
    io.write_string("main\n", !IO).

% MERK-DAG: merk.initialise @init_directives.first_2_0{{$}}
% MERK-DAG: merk.initialise @init_directives.last_0_0 final
% MERK-DAG: merk.initialise @init_directives.second_0_0{{$}}

% LLVM: @llvm.global_ctors = {{.*}} @merk.register_initialisers
% LLVM: define internal void @merk.register_initialisers()
% LLVM-NEXT: call void @merk_register_initialiser(ptr @init_directives.first_2_0.init, i32 0)
% LLVM-NEXT: call void @merk_register_initialiser(ptr @init_directives.last_0_0.init, i32 1)
% LLVM-NEXT: call void @merk_register_initialiser(ptr @init_directives.second_0_0.init, i32 0)

% OUTPUT:first
% OUTPUT-NEXT:second
% OUTPUT-NEXT:main
% OUTPUT-NEXT:last
