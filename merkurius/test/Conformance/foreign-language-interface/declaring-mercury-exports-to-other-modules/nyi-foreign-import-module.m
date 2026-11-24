% MANUAL: Declaring Mercury exports to other modules
% foreign_import_module makes another module's foreign_export declarations
% visible to the module's foreign code, which calls the function it exports.
% MODULES: Inputs/exporting.m
%
% RUN: not merkc -I %S/Inputs %s %S/Inputs/exporting.m -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: pragma foreign_import_module of a module with foreign_exports

:- module nyi_foreign_import_module.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module exporting.
:- pragma foreign_import_module("C", exporting).
:- pred call_triple(int::in, int::out) is det.
:- pragma foreign_proc("C", call_triple(X::in, Y::out),
    [promise_pure, may_call_mercury], "Y = exporting_triple(X);").

main(!IO) :-
    call_triple(4, Y), io.write_int(Y, !IO), io.nl(!IO),
    io.write_int(triple(2), !IO), io.nl(!IO).
