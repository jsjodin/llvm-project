% The module loader reads the interfaces of the modules a module imports,
% from their sources: the library's from the library directory, others'
% from the module's directory and the -I directories. Each imported
% interface's own imports are loaded too, its implementation's are not,
% and every module imports builtin.
%
% RUN: merkc -fc1 -fdebug-dump-imports -I %S/Inputs %s 2>&1 | FileCheck --implicit-check-not=error %s

:- module imports.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module helper, list.
:- use_module string.

main(!IO) :-
    io.write_string(string.from_int(helper.first_or([1, 2], 0)), !IO).

% CHECK:      imports: import_module io helper list builtin use_module string
% CHECK-NEXT: io (io.m, {{[0-9]+}} items): import_module char list string univ builtin
% CHECK-NEXT: char (char.m, {{[0-9]+}} items): import_module list builtin
% CHECK-NEXT: list (list.m, {{[0-9]+}} items): import_module builtin
% CHECK-NEXT: builtin (builtin.m, {{[0-9]+}} items):
% CHECK-NEXT: string (string.m, {{[0-9]+}} items): import_module char list builtin
% CHECK-NEXT: univ (univ.m, {{[0-9]+}} items): import_module builtin
% CHECK-NEXT: helper (helper.m, 2 items): import_module list builtin use_module maybe
% CHECK-NEXT: maybe (maybe.m, {{[0-9]+}} items): import_module list builtin
% CHECK-NOT:  {{.}}
