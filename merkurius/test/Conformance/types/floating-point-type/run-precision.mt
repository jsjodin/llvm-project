% MANUAL: Floating-point type
% DESIGN: Sized floating point types
% float is IEEE 754 single or double precision, the implementation's choice;
% here it is double, as in mmc's default grades: its epsilon, extremes and
% rounding are double's. float32 is single precision, so its are single's:
% 2^53 + 1 rounds to 2^53 in both, and 2^24 + 1 to 2^24 in float32 only.
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 %t.float.d/run_precision_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/run_precision_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:2.220446049250313e-16
% OUTPUT-FLOAT-NEXT:1.7976931348623157e+308
% OUTPUT-FLOAT-NEXT:2.2250738585072014e-308
% OUTPUT-FLOAT-NEXT:0.30000000000000004
% OUTPUT-FLOAT-NEXT:0.3333333333333333
% OUTPUT-FLOAT-NEXT:9007199254740992.0
% OUTPUT-FLOAT-NEXT:16777217.0
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 %t.float32.d/run_precision_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/run_precision_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1.1920929e-07f32
% OUTPUT-FLOAT32-NEXT:3.4028235e+38f32
% OUTPUT-FLOAT32-NEXT:1.1754944e-38f32
% OUTPUT-FLOAT32-NEXT:0.3f32
% OUTPUT-FLOAT32-NEXT:0.33333334f32
% OUTPUT-FLOAT32-NEXT:9.007199e+15f32
% OUTPUT-FLOAT32-NEXT:16777216.0f32
% OUTPUT-FLOAT32-NOT:{{.}}

:- module run_precision.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int.

main(!IO) :-
    io.write_line(@T@.epsilon : @T@, !IO),
    io.write_line(@T@.max : @T@, !IO),
    io.write_line(@T@.min : @T@, !IO),
    io.write_line(a + b, !IO),
    io.write_line(1.0@S@ / 3.0@S@, !IO),
    io.write_line(@FROM_INT@(int.(1 << 53)) + 1.0@S@, !IO),
    io.write_line(@FROM_INT@(int.(1 << 24)) + 1.0@S@, !IO).

:- func a = @T@.
:- pragma no_inline(a/0).
a = 0.1@S@.
:- func b = @T@.
:- pragma no_inline(b/0).
b = 0.2@S@.
