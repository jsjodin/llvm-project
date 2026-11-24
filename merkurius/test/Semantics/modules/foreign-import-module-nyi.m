% A foreign_import_module of a module that the program imports makes that
% module's foreign_exports and shared foreign_decls visible to the module's
% foreign code (reference manual, "Declaring Mercury exports to other
% modules"); one of a module it does not import would need that module
% loaded for it alone, which is not yet implemented.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>&1 >/dev/null \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module foreign_import_module_nyi.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pragma foreign_import_module("C", io).
:- pragma foreign_import_module("C", bool).
% CHECK: foreign-import-module-nyi.m:[[@LINE-1]]:4: error: not yet implemented: pragma foreign_import_module of a module that is not imported
main(!IO) :- io.write_string("main\n", !IO).
