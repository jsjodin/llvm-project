% MANUAL: Using foreign enumerations in Mercury code
% foreign_enum gives an enumeration's constructors the values of C
% constants, which a foreign_proc sees.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: pragma foreign_enum

:- module nyi_foreign_enum.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.
:- type fruit ---> apple ; banana ; cherry.
:- pragma foreign_decl("C", local, "#define VA 7\n#define VB 3\n#define VC 5").
:- pragma foreign_enum("C", fruit/0,
    [apple - "VA", banana - "VB", cherry - "VC"]).
:- func value(fruit) = int.
:- pragma foreign_proc("C", value(T::in) = (V::out),
    [promise_pure, will_not_call_mercury], "V = T;").

main(!IO) :-
    io.write([value(apple), value(banana), value(cherry)], !IO), io.nl(!IO),
    compare(R, banana, apple), io.write(R, !IO), io.nl(!IO).
