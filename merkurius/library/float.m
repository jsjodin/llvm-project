% A stub of the standard library module float, for name resolution until
% the library is written, with its functions implemented but hash. The
% declarations are those of the library reference manual (22.01), less
% those that need modules not stubbed yet; see README.md. The arithmetic
% and comparisons are built into code generation.

:- module float.
:- interface.

:- func (float::in) + (float::in) = (float::uo) is det.
:- func (float::in) - (float::in) = (float::uo) is det.
:- func (float::in) * (float::in) = (float::uo) is det.
:- func (float::in) / (float::in) = (float::uo) is det.
:- func unchecked_quotient(float::in, float::in) = (float::uo) is det.
:- func + (float::in) = (float::uo) is det.
:- func - (float::in) = (float::uo) is det.
:- pred (float::in) < (float::in) is semidet.
:- pred (float::in) =< (float::in) is semidet.
:- pred (float::in) >= (float::in) is semidet.
:- pred (float::in) > (float::in) is semidet.
:- func float(int) = float.
:- func from_int8(int8) = float.
:- func from_int16(int16) = float.
:- func cast_from_int32(int32) = float.
:- func cast_from_int64(int64) = float.
:- func from_uint8(uint8) =  float.
:- func from_uint16(uint16) = float.
:- func cast_from_uint32(uint32) = float.
:- func cast_from_uint64(uint64) = float.
:- func ceiling_to_int(float) = int.
:- func floor_to_int(float) = int.
:- func round_to_int(float) = int.
:- func truncate_to_int(float) = int.
:- func abs(float) = float.
:- func max(float, float) = float.
:- func min(float, float) = float.
:- func pow(float, int) = float.
:- func hash(float) = int.
:- pred hash(float::in, int::out) is det.
:- pred is_infinite(float::in) is semidet.
:- pred is_inf(float::in) is semidet.
:- pred is_nan(float::in) is semidet.
:- pred is_nan_or_infinite(float::in) is semidet.
:- pred is_nan_or_inf(float::in) is semidet.
:- pred is_finite(float::in) is semidet.
:- pred is_zero(float::in) is semidet.
:- func max = float.
:- func min = float.
:- func infinity = float.
:- func epsilon = float.
:- func radix = int.
:- func mantissa_digits = int.
:- func min_exponent = int.
:- func max_exponent = int.

:- implementation.
:- import_module int.
:- import_module int16.
:- import_module int32.
:- import_module int64.
:- import_module int8.
:- import_module uint16.
:- import_module uint32.
:- import_module uint8.

:- pragma foreign_decl("C", local, "
#include <float.h>
#include <math.h>

/* Until exceptions are implemented, nothing catches one. */
void merk_uncaught_exception(const char *exception)
    __attribute__((noreturn));
").

from_int8(N) = float(int8.to_int(N)).
from_int16(N) = float(int16.to_int(N)).
cast_from_int32(N) = float(int32.to_int(N)).
cast_from_int64(N) = float(int64.cast_to_int(N)).
from_uint8(N) = float(uint8.to_int(N)).
from_uint16(N) = float(uint16.to_int(N)).
cast_from_uint32(N) = float(uint32.cast_to_int(N)).

:- pragma foreign_proc("C",
    cast_from_uint64(N::in) = (F::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    F = (MR_Float) N;
").

% A float out of the range of int converts to min_int, as mmc's programs
% do on x86-64; X rounds to floor(X + 0.5).
:- pragma foreign_proc("C",
    ceiling_to_int(F::in) = (N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    MR_Float x = ceil(F);
    N = x >= -0x1p63 && x < 0x1p63 ? (MR_Integer) x : INTPTR_MIN;
").

:- pragma foreign_proc("C",
    floor_to_int(F::in) = (N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    MR_Float x = floor(F);
    N = x >= -0x1p63 && x < 0x1p63 ? (MR_Integer) x : INTPTR_MIN;
").

:- pragma foreign_proc("C",
    round_to_int(F::in) = (N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    MR_Float x = floor(F + 0.5);
    N = x >= -0x1p63 && x < 0x1p63 ? (MR_Integer) x : INTPTR_MIN;
").

:- pragma foreign_proc("C",
    truncate_to_int(F::in) = (N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    MR_Float x = F;
    N = x >= -0x1p63 && x < 0x1p63 ? (MR_Integer) x : INTPTR_MIN;
").

% max and min are those of the comparisons: max(X, Y) is X if X >= Y, and
% min(X, Y) is X if X =< Y, else Y, which decide NaN and -0.0 as mmc's
% programs do.
:- pragma foreign_proc("C",
    abs(X::in) = (A::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    A = fabs(X);
").

:- pragma foreign_proc("C",
    max(X::in, Y::in) = (M::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    M = X >= Y ? X : Y;
").

:- pragma foreign_proc("C",
    min(X::in, Y::in) = (M::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    M = X <= Y ? X : Y;
").

% pow multiplies by squaring, from the exponent's low bits, as mmc's
% programs round; a negative exponent is that of the reciprocal.
:- pragma foreign_proc("C",
    pow(X::in, N::in) = (P::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    if (X == 0.0 && N < 0) {
        merk_uncaught_exception(\"domain_error(\\\"float.pow: zero base\\\")\");
    }
    MR_Integer n = N < 0 ? -N : N;
    MR_Float x = X, p = 1.0;
    while (n > 0) {
        if (n & 1) {
            p *= x;
        }
        x *= x;
        n >>= 1;
    }
    P = N < 0 ? 1.0 / p : p;
").

:- pragma foreign_proc("C",
    is_infinite(X::in),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = isinf(X);
").

:- pragma foreign_proc("C",
    is_inf(X::in),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = isinf(X);
").

:- pragma foreign_proc("C",
    is_nan(X::in),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = isnan(X);
").

:- pragma foreign_proc("C",
    is_nan_or_infinite(X::in),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = !isfinite(X);
").

:- pragma foreign_proc("C",
    is_nan_or_inf(X::in),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = !isfinite(X);
").

:- pragma foreign_proc("C",
    is_finite(X::in),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = isfinite(X);
").

:- pragma foreign_proc("C",
    is_zero(X::in),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = X == 0.0;
").

:- pragma foreign_proc("C",
    max = (F::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    F = DBL_MAX;
").

:- pragma foreign_proc("C",
    min = (F::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    F = DBL_MIN;
").

:- pragma foreign_proc("C",
    infinity = (F::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    F = INFINITY;
").

:- pragma foreign_proc("C",
    epsilon = (F::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    F = DBL_EPSILON;
").

:- pragma foreign_proc("C",
    radix = (N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    N = FLT_RADIX;
").

:- pragma foreign_proc("C",
    mantissa_digits = (N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    N = DBL_MANT_DIG;
").

:- pragma foreign_proc("C",
    min_exponent = (N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    N = DBL_MIN_EXP;
").

:- pragma foreign_proc("C",
    max_exponent = (N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    N = DBL_MAX_EXP;
").
