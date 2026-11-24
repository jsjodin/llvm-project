% MANUAL: Using foreign types from Mercury
% A discriminated union type whose foreign_type for C replaces its
% Mercury definition when compiling to C.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: pragma foreign_type for a discriminated union type

:- module nyi_discriminated_union.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t ---> t(int).
:- pragma foreign_type("C", t, "long").
:- func mk(int) = t.
:- pragma foreign_proc("C", mk(X::in) = (T::out),
    [promise_pure, will_not_call_mercury], "T = X * 2;").
:- func get(t) = int.
:- pragma foreign_proc("C", get(T::in) = (X::out),
    [promise_pure, will_not_call_mercury], "X = T;").

main(!IO) :- io.write_int(get(mk(21)), !IO), io.nl(!IO).
