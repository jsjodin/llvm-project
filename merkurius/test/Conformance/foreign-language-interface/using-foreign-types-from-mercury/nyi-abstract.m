% MANUAL: Using foreign types from Mercury
% An abstract type defined as a C struct pointer by foreign_type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for values of type nyi_abstract.point

:- module nyi_abstract.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type point.
:- pragma foreign_type("C", point, "struct point *").
:- pragma foreign_decl("C", local,
    "struct point { long x, y; };\n#include <stdlib.h>").
:- func mk(int, int) = point.
:- pragma foreign_proc("C", mk(X::in, Y::in) = (P::out),
    [promise_pure, will_not_call_mercury],
    "P = malloc(sizeof(struct point)); P->x = X; P->y = Y;").
:- func px(point) = int.
:- pragma foreign_proc("C", px(P::in) = (X::out),
    [promise_pure, will_not_call_mercury], "X = P->x;").

main(!IO) :- P = mk(3, 4), io.write_int(px(P), !IO), io.nl(!IO).
