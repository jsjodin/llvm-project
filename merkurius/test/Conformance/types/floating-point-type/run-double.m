% MANUAL: Floating-point type
% float is IEEE 754 single or double precision, the implementation's choice;
% here it is double, as in mmc's default grades: its epsilon, extremes and
% rounding are double's.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:2.220446049250313e-16
% OUTPUT-NEXT:1.7976931348623157e+308
% OUTPUT-NEXT:2.2250738585072014e-308
% OUTPUT-NEXT:0.30000000000000004
% OUTPUT-NEXT:0.3333333333333333
% OUTPUT-NEXT:9007199254740992.0
% OUTPUT-NOT:{{.}}

:- module run_double.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int.

main(!IO) :-
    io.write_line(float.epsilon : float, !IO),
    io.write_line(float.max : float, !IO),
    io.write_line(float.min : float, !IO),
    io.write_line(a + b, !IO),
    io.write_line(1.0 / 3.0, !IO),
    io.write_line(float.float(int.(1 << 53)) + 1.0, !IO).

:- func a = float.
:- pragma no_inline(a/0).
a = 0.1.
:- func b = float.
:- pragma no_inline(b/0).
b = 0.2.
