% The library module float32, a Merkurius extension (docs/INPROGRESS.md,
% Sized floating point types): IEEE single precision. Its arithmetic,
% comparisons and conversions to and from float are built into code
% generation, as float's are; the rest are C's, on float. Not in the
% library reference manual or in mmc.

:- module float32.
:- interface.

:- func (float32::in) + (float32::in) = (float32::uo) is det.
:- func (float32::in) - (float32::in) = (float32::uo) is det.
:- func (float32::in) * (float32::in) = (float32::uo) is det.
:- func (float32::in) / (float32::in) = (float32::uo) is det.
:- func unchecked_quotient(float32::in, float32::in) = (float32::uo) is det.
:- func + (float32::in) = (float32::uo) is det.
:- func - (float32::in) = (float32::uo) is det.
:- pred (float32::in) < (float32::in) is semidet.
:- pred (float32::in) =< (float32::in) is semidet.
:- pred (float32::in) >= (float32::in) is semidet.
:- pred (float32::in) > (float32::in) is semidet.

    % The float rounded to the nearest float32, ties to even; out of its
    % range, an infinity.
    %
:- func from_float(float) = float32.

    % The float32 as a float, exactly.
    %
:- func to_float(float32) = float.

    % The int rounded to the nearest float32, ties to even.
    %
:- func cast_from_int(int) = float32.

:- func ceiling_to_int(float32) = int.
:- func floor_to_int(float32) = int.
:- func round_to_int(float32) = int.
:- func truncate_to_int(float32) = int.
:- func abs(float32) = float32.
:- func max(float32, float32) = float32.
:- func min(float32, float32) = float32.
:- func pow(float32, int) = float32.
:- pred is_infinite(float32::in) is semidet.
:- pred is_inf(float32::in) is semidet.
:- pred is_nan(float32::in) is semidet.
:- pred is_nan_or_infinite(float32::in) is semidet.
:- pred is_nan_or_inf(float32::in) is semidet.
:- pred is_finite(float32::in) is semidet.
:- pred is_zero(float32::in) is semidet.
:- func max = float32.
:- func min = float32.
:- func infinity = float32.
:- func epsilon = float32.
:- func radix = int.
:- func mantissa_digits = int.
:- func min_exponent = int.
:- func max_exponent = int.

:- implementation.

:- pragma foreign_decl("C", local, "
#include <float.h>
#include <math.h>

/* An error of the library's, with its text as mmc reports it uncaught,
   which the runtime throws as its value (runtime/exception.cpp). A GPU's
   runtime reports it, under its own name, as nothing catches it there. */
#ifdef __AMDGPU__
void merk_device_uncaught_exception(const char *exception)
    __attribute__((noreturn));
#define merk_library_error merk_device_uncaught_exception
#else
void merk_library_error(const char *exception) __attribute__((noreturn));
#endif
").

% A float32 out of the range of int converts to min_int, as float's do;
% X rounds to floor(X + 0.5), in single precision.
:- pragma foreign_proc("C",
    ceiling_to_int(F::in) = (N::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    MR_Float32 x = ceilf(F);
    N = x >= -0x1p63f && x < 0x1p63f ? (MR_Integer) x : INTPTR_MIN;
").

:- pragma foreign_proc("C",
    floor_to_int(F::in) = (N::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    MR_Float32 x = floorf(F);
    N = x >= -0x1p63f && x < 0x1p63f ? (MR_Integer) x : INTPTR_MIN;
").

:- pragma foreign_proc("C",
    round_to_int(F::in) = (N::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    MR_Float32 x = floorf(F + 0.5f);
    N = x >= -0x1p63f && x < 0x1p63f ? (MR_Integer) x : INTPTR_MIN;
").

:- pragma foreign_proc("C",
    truncate_to_int(F::in) = (N::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    MR_Float32 x = F;
    N = x >= -0x1p63f && x < 0x1p63f ? (MR_Integer) x : INTPTR_MIN;
").

% max and min are those of the comparisons, as float's are.
:- pragma foreign_proc("C",
    abs(X::in) = (A::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    A = fabsf(X);
").

:- pragma foreign_proc("C",
    max(X::in, Y::in) = (M::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    M = X >= Y ? X : Y;
").

:- pragma foreign_proc("C",
    min(X::in, Y::in) = (M::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    M = X <= Y ? X : Y;
").

% pow multiplies by squaring, in single precision, as float.pow does in
% double.
:- pragma foreign_proc("C",
    pow(X::in, N::in) = (P::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    if (X == 0.0f && N < 0) {
        merk_library_error(\"domain_error(\\\"float32.pow: zero base\\\")\");
    }
    MR_Integer n = N < 0 ? -N : N;
    MR_Float32 x = X, p = 1.0f;
    while (n > 0) {
        if (n & 1) {
            p *= x;
        }
        x *= x;
        n >>= 1;
    }
    P = N < 0 ? 1.0f / p : p;
").

:- pragma foreign_proc("C",
    is_infinite(X::in),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = isinf(X);
").

:- pragma foreign_proc("C",
    is_inf(X::in),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = isinf(X);
").

:- pragma foreign_proc("C",
    is_nan(X::in),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = isnan(X);
").

:- pragma foreign_proc("C",
    is_nan_or_infinite(X::in),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = !isfinite(X);
").

:- pragma foreign_proc("C",
    is_nan_or_inf(X::in),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = !isfinite(X);
").

:- pragma foreign_proc("C",
    is_finite(X::in),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = isfinite(X);
").

:- pragma foreign_proc("C",
    is_zero(X::in),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = X == 0.0f;
").

:- pragma foreign_proc("C",
    max = (F::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    F = FLT_MAX;
").

:- pragma foreign_proc("C",
    min = (F::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    F = FLT_MIN;
").

:- pragma foreign_proc("C",
    infinity = (F::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    F = INFINITY;
").

:- pragma foreign_proc("C",
    epsilon = (F::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    F = FLT_EPSILON;
").

:- pragma foreign_proc("C",
    radix = (N::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    N = FLT_RADIX;
").

:- pragma foreign_proc("C",
    mantissa_digits = (N::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    N = FLT_MANT_DIG;
").

:- pragma foreign_proc("C",
    min_exponent = (N::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    N = FLT_MIN_EXP;
").

:- pragma foreign_proc("C",
    max_exponent = (N::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    N = FLT_MAX_EXP;
").
