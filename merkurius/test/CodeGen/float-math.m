% The float and math modules of the compiled standard library, whose
% functions are foreign_procs calling C's math library, and float's
% arithmetic, which is built in. As in mmc's programs, -X is 0.0 - X, so
% -0.0 is 0.0; max and min are those of >= and =<; round adds 0.5 and takes
% the floor; float.pow multiplies by squaring; a float out of int's range
% converts to min_int; math's domain restrictions throw domain_error. The
% output is mmc's for the same program, which ends with sqrt(-1.0).
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc %s -o %t
% RUN: not %t > %t.out 2> %t.err
% RUN: FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace < %t.out
% RUN: FileCheck %s --check-prefix=ERROR --match-full-lines --strict-whitespace < %t.err
% RUN: merkc -O2 %s -o %t.O2
% RUN: not %t.O2 > %t.O2.out 2> %t.O2.err
% RUN: FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace < %t.O2.out
% RUN: FileCheck %s --check-prefix=ERROR --match-full-lines --strict-whitespace < %t.O2.err

% MERK-LABEL: merk.proc @float_math.main_2_0()
% MERK: %[[Z:.*]] = merk.call @float_math.zero_1_0() : () -> f64
% MERK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
% MERK-NEXT: %{{.*}} = arith.subf %[[ZERO]], %[[Z]] : f64
% MERK: merk.call @math.sqrt_f1_0(%{{.*}}) : (f64) -> f64

% OUTPUT:0.0 finite zero
% OUTPUT-NEXT:-0.0 finite zero
% OUTPUT-NEXT:0.0 finite zero
% OUTPUT-NEXT:-0.0 finite zero
% OUTPUT-NEXT:0.0 finite zero
% OUTPUT-NEXT:-0.0 finite zero
% OUTPUT-NEXT:0.0 finite zero
% OUTPUT-NEXT:1.0 finite
% OUTPUT-NEXT:nan nan nan_or_inf
% OUTPUT-NEXT:1.0 finite
% OUTPUT-NEXT:nan nan nan_or_inf
% OUTPUT-NEXT:8.954302432552389 finite
% OUTPUT-NEXT:1.1051710231252438 finite
% OUTPUT-NEXT:0.5131581182307065 finite
% OUTPUT-NEXT:1.0 finite
% OUTPUT-NEXT:-2.0 finite
% OUTPUT-NEXT:1.0 finite
% OUTPUT-NEXT:-9.223372036854776e+18 finite
% OUTPUT-NEXT:0.0 finite zero
% OUTPUT-NEXT:-1.0 finite
% OUTPUT-NEXT:infinity inf nan_or_inf
% OUTPUT-NEXT:1.7976931348623157e+308 finite
% OUTPUT-NEXT:2.2250738585072014e-308 finite
% OUTPUT-NEXT:infinity inf nan_or_inf
% OUTPUT-NEXT:2.220446049250313e-16 finite
% OUTPUT-NEXT:2 53 -1021 1024
% OUTPUT-NEXT:-128.0 finite
% OUTPUT-NEXT:1.8446744073709552e+19 finite
% OUTPUT-NEXT:4294967295.0 finite
% OUTPUT-NEXT:3.141592653589793 finite
% OUTPUT-NEXT:2.718281828459045 finite
% OUTPUT-NEXT:-0.0 finite zero
% OUTPUT-NEXT:-1.0 finite
% OUTPUT-NEXT:-2.0 finite
% OUTPUT-NEXT:3.0 finite
% OUTPUT-NEXT:0.0 finite zero
% OUTPUT-NEXT:1.0 finite
% OUTPUT-NEXT:-1.0 finite
% OUTPUT-NEXT:1.4142135623730951 finite
% OUTPUT-NEXT:1.4142135623730951 finite
% OUTPUT-NEXT:2.718281828459045 finite
% OUTPUT-NEXT:2.302585092994046 finite
% OUTPUT-NEXT:3.0 finite
% OUTPUT-NEXT:10.0 finite
% OUTPUT-NEXT:4.0 finite
% OUTPUT-NEXT:0.8414709848078965 finite
% OUTPUT-NEXT:0.5403023058681398 finite
% OUTPUT-NEXT:1.5574077246549023 finite
% OUTPUT-NEXT:0.5235987755982989 finite
% OUTPUT-NEXT:1.0471975511965979 finite
% OUTPUT-NEXT:0.7853981633974483 finite
% OUTPUT-NEXT:2.356194490192345 finite
% OUTPUT-NEXT:1.1752011936438014 finite
% OUTPUT-NEXT:1.5430806348152437 finite
% OUTPUT-NEXT:0.7615941559557649 finite
% OUTPUT-NEXT:7.0 finite
% OUTPUT-NOT:{{.}}

% ERROR:Uncaught Mercury exception:
% ERROR-NEXT:domain_error("math.sqrt")
% ERROR-NOT:{{.}}

:- module float_math.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, int8, math, uint32, uint64.
:- pred zero(float::out) is det.
:- pragma no_inline(zero/1).
zero(0.0).
:- pred w(float::in, io::di, io::uo) is det.
w(X, !IO) :-
    io.write_float(X, !IO),
    ( if is_nan(X) then io.write_string(" nan", !IO) else true ),
    ( if is_inf(X) then io.write_string(" inf", !IO) else true ),
    ( if is_finite(X) then io.write_string(" finite", !IO) else true ),
    ( if is_nan_or_inf(X) then io.write_string(" nan_or_inf", !IO) else true ),
    ( if is_zero(X) then io.write_string(" zero", !IO) else true ),
    io.nl(!IO).
main(!IO) :-
    zero(Z),
    NaN = exp(Z + 1000.0) * Z,
    MinusZero = Z * -1.0,
    w(-Z, !IO), w(MinusZero, !IO), w(abs(MinusZero), !IO),
    w(max(MinusZero, Z), !IO), w(max(Z, MinusZero), !IO),
    w(min(MinusZero, Z), !IO), w(min(Z, MinusZero), !IO),
    w(max(NaN, 1.0), !IO), w(max(1.0, NaN), !IO),
    w(min(NaN, 1.0), !IO), w(min(1.0, NaN), !IO),
    w(float.pow(Z + 1.1, 23), !IO), w(float.pow(Z + 1.0000001, 1000001), !IO),
    w(float.pow(Z + 1.1, -7), !IO), w(float.pow(Z, 0), !IO),
    w(float(round_to_int(Z - 2.5)), !IO), w(float(round_to_int(Z + 0.5)), !IO),
    w(float(floor_to_int(Z + 1.0e30)), !IO), w(float(ceiling_to_int(Z - 0.5)), !IO),
    w(float(truncate_to_int(Z - 1.7)), !IO),
    w(unchecked_quotient(Z + 1.0, Z), !IO),
    w(float.max, !IO), w(float.min, !IO), w(infinity, !IO), w(epsilon, !IO),
    io.write_int(radix, !IO), io.write_char(' ', !IO),
    io.write_int(mantissa_digits, !IO), io.write_char(' ', !IO),
    io.write_int(min_exponent, !IO), io.write_char(' ', !IO),
    io.write_int(max_exponent, !IO), io.nl(!IO),
    w(from_int8(int8.min_int8), !IO), w(cast_from_uint64(uint64.max_uint64), !IO),
    w(cast_from_uint32(uint32.max_uint32), !IO),
    w(pi, !IO), w(e, !IO),
    w(ceiling(Z - 0.5), !IO), w(floor(Z - 0.5), !IO),
    w(round(Z - 2.5), !IO), w(round(Z + 2.5), !IO), w(round(Z - 0.3), !IO),
    w(round(Z + 0.49999999999999994), !IO), w(truncate(Z - 1.7), !IO),
    w(sqrt(Z + 2.0), !IO), w(math.pow(Z + 2.0, 0.5), !IO), w(exp(Z + 1.0), !IO),
    w(ln(Z + 10.0), !IO), w(log10(Z + 1000.0), !IO), w(log2(Z + 1024.0), !IO),
    w(log(Z + 3.0, 81.0), !IO),
    w(sin(Z + 1.0), !IO), w(cos(Z + 1.0), !IO), w(tan(Z + 1.0), !IO),
    w(asin(Z + 0.5), !IO), w(acos(Z + 0.5), !IO), w(atan(Z + 1.0), !IO),
    w(atan2(Z + 1.0, Z - 1.0), !IO),
    w(sinh(Z + 1.0), !IO), w(cosh(Z + 1.0), !IO), w(tanh(Z + 1.0), !IO),
    ( if have_fma then w(fma(Z + 2.0, 3.0, 1.0), !IO) else true ),
    w(sqrt(Z - 1.0), !IO).
