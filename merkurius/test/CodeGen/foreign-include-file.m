% include_file(Path) in foreign_decl and foreign_code: a relative path is
% relative to the directory of the module's source, not to where merkc runs
% nor to where it compiles the foreign code (reference manual, "Adding
% foreign declarations" and "Adding foreign definitions").
%
% RUN: rm -rf %t && mkdir %t && cd %t
% RUN: merkc %s -o prog && ./prog | FileCheck %s --match-full-lines
% CHECK:11

:- module foreign_include_file.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- pragma foreign_decl("C", include_file("Inputs/foreign-include/values.h")).
:- pragma foreign_code("C", include_file("Inputs/foreign-include/functions.c")).

:- func value = int.
:- pragma foreign_proc("C", value = (V::out),
    [promise_pure, will_not_call_mercury],
    "V = INCLUDED_VALUE + included_function();").

main(!IO) :-
    io.write_int(value, !IO),
    io.nl(!IO).
