% MANUAL: Fact tables
% A fact table with a semidet mode, its facts in a file.
% FILES: Inputs/ages
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: fact tables (`:- pragma fact_table')

:- module nyi_semidet.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred age(string::in, int::out) is semidet.
:- pragma fact_table(age/2, "ages").

main(!IO) :-
    ( if age("bob", A) then io.write_int(A, !IO) else io.write_string("none", !IO) ),
    io.nl(!IO),
    ( if age("dave", B) then io.write_int(B, !IO) else io.write_string("none", !IO) ),
    io.nl(!IO).
