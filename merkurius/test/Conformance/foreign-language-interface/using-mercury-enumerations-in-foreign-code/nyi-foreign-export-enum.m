% MANUAL: Using Mercury enumerations in foreign code
% foreign_export_enum gives C names to an enumeration's constructors, with
% a prefix, upper case, and an override.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: pragma foreign_export_enum

:- module nyi_foreign_export_enum.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t ---> red ; green ; blue.
:- pragma foreign_export_enum("C", t/0, [prefix("COLOUR_"), uppercase],
    [blue - "AZURE"]).
:- func azure = t.
:- pragma foreign_proc("C", azure = (T::out),
    [promise_pure, will_not_call_mercury], "T = COLOUR_AZURE;").
:- func leaf = t.
:- pragma foreign_proc("C", leaf = (T::out),
    [promise_pure, will_not_call_mercury], "T = COLOUR_GREEN;").

main(!IO) :- io.write({azure, leaf}, !IO), io.nl(!IO).
