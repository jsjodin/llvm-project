% MANUAL: Using Mercury enumerations in foreign code
% The uppercase attribute given twice, which the manual does not forbid, as
% it does a second prefix, and which mmc accepts.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: pragma foreign_export_enum

:- module nyi_uppercase_twice.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t ---> red ; green ; blue.
:- pragma foreign_export_enum("C", t/0, [uppercase, prefix("C_"), uppercase]).
:- func leaf = t.
:- pragma foreign_proc("C", leaf = (T::out),
    [promise_pure, will_not_call_mercury], "T = C_GREEN;").

main(!IO) :- io.write(leaf, !IO), io.nl(!IO).
