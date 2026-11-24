% MANUAL: Calling Mercury from foreign code
% A det function exported to C, which a foreign_proc that may call
% Mercury calls.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: pragma foreign_export

:- module nyi_foreign_export.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.
:- func triple(int) = int.
triple(X) = 3 * X.
:- pragma foreign_export("C", triple(in) = out, "triple_from_c").
:- pred call_triple(int::in, int::out) is det.
:- pragma foreign_proc("C", call_triple(X::in, Y::out),
    [promise_pure, may_call_mercury], "Y = triple_from_c(X);").

main(!IO) :- call_triple(4, Y), io.write_int(Y, !IO), io.nl(!IO).
