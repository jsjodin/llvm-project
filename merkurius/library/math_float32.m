% math's functions on float32, a Merkurius extension (docs/INPROGRESS.md,
% Sized floating point types), under math's names, so that a program that
% imports both calls the one of its argument's type. They are implemented
% with C's single-precision functions, sqrtf and so on, and check the same
% domains as math's, throwing domain_error("math_float32.Name"). math's
% solve_quadratic has no float32 form.

:- module math_float32.
:- interface.
:- import_module float32.

:- func pi = float32.
:- func e = float32.
:- func ceiling(float32) = float32.
:- func floor(float32) = float32.
:- func round(float32) = float32.
:- func truncate(float32) = float32.
:- func sqrt(float32) = float32.
:- func unchecked_sqrt(float32) = float32.
:- func pow(float32, float32) = float32.
:- func unchecked_pow(float32, float32) = float32.
:- func exp(float32) = float32.
:- func ln(float32) = float32.
:- func unchecked_ln(float32) = float32.
:- func log10(float32) = float32.
:- func unchecked_log10(float32) = float32.
:- func log2(float32) = float32.
:- func unchecked_log2(float32) = float32.
:- func log(float32, float32) = float32.
:- func unchecked_log(float32, float32) = float32.
:- func sin(float32) = float32.
:- func cos(float32) = float32.
:- func tan(float32) = float32.
:- func asin(float32) = float32.
:- func unchecked_asin(float32) = float32.
:- func acos(float32) = float32.
:- func unchecked_acos(float32) = float32.
:- func atan(float32) = float32.
:- func atan2(float32, float32) = float32.
:- func sinh(float32) = float32.
:- func cosh(float32) = float32.
:- func tanh(float32) = float32.
:- pred have_fma is semidet.
:- func fma(float32, float32, float32) = float32.

:- implementation.

:- pragma foreign_decl("C", local, "
#include <math.h>
#include <stdio.h>

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

static void merk_domain_error(const char *what)
{
    char buffer[256];
    snprintf(buffer, sizeof buffer, \"domain_error(\\\"%s\\\")\", what);
    merk_library_error(buffer);
}
").

:- pragma foreign_proc("C",
    pi = (P::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    P = 3.14159265358979323846f;
").

:- pragma foreign_proc("C",
    e = (E::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    E = 2.7182818284590452354f;
").

% round(X) is floor(X + 0.5f): a fraction of 0.5f rounds up.
:- pragma foreign_proc("C",
    ceiling(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = ceilf(X);
").

:- pragma foreign_proc("C",
    floor(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = floorf(X);
").

:- pragma foreign_proc("C",
    round(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = floorf(X + 0.5f);
").

:- pragma foreign_proc("C",
    truncate(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = truncf(X);
").

:- pragma foreign_proc("C",
    sqrt(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    if (X < 0.0f) {
        merk_domain_error(\"math_float32.sqrt\");
    }
    Y = sqrtf(X);
").

:- pragma foreign_proc("C",
    unchecked_sqrt(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = sqrtf(X);
").

:- pragma foreign_proc("C",
    pow(X::in, P::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    if (X < 0.0f || (X == 0.0f && P <= 0.0f)) {
        merk_domain_error(\"math_float32.pow\");
    }
    Y = powf(X, P);
").

:- pragma foreign_proc("C",
    unchecked_pow(X::in, P::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = powf(X, P);
").

:- pragma foreign_proc("C",
    ln(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    if (X <= 0.0f) {
        merk_domain_error(\"math_float32.ln\");
    }
    Y = logf(X);
").

:- pragma foreign_proc("C",
    unchecked_ln(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = logf(X);
").

:- pragma foreign_proc("C",
    log10(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    if (X <= 0.0f) {
        merk_domain_error(\"math_float32.log10\");
    }
    Y = log10f(X);
").

:- pragma foreign_proc("C",
    unchecked_log10(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = log10f(X);
").

:- pragma foreign_proc("C",
    log2(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    if (X <= 0.0f) {
        merk_domain_error(\"math_float32.log2\");
    }
    Y = log2f(X);
").

:- pragma foreign_proc("C",
    unchecked_log2(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = log2f(X);
").

:- pragma foreign_proc("C",
    log(B::in, X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    if (X <= 0.0f || B <= 0.0f || B == 1.0f) {
        merk_domain_error(\"math_float32.log\");
    }
    Y = logf(X) / logf(B);
").

:- pragma foreign_proc("C",
    unchecked_log(B::in, X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = logf(X) / logf(B);
").

:- pragma foreign_proc("C",
    asin(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    if (X < -1.0f || X > 1.0f) {
        merk_domain_error(\"math_float32.asin\");
    }
    Y = asinf(X);
").

:- pragma foreign_proc("C",
    unchecked_asin(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = asinf(X);
").

:- pragma foreign_proc("C",
    acos(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    if (X < -1.0f || X > 1.0f) {
        merk_domain_error(\"math_float32.acos\");
    }
    Y = acosf(X);
").

:- pragma foreign_proc("C",
    unchecked_acos(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = acosf(X);
").

:- pragma foreign_proc("C",
    exp(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = expf(X);
").

:- pragma foreign_proc("C",
    sin(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = sinf(X);
").

:- pragma foreign_proc("C",
    cos(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = cosf(X);
").

:- pragma foreign_proc("C",
    tan(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = tanf(X);
").

:- pragma foreign_proc("C",
    atan(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = atanf(X);
").

:- pragma foreign_proc("C",
    sinh(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = sinhf(X);
").

:- pragma foreign_proc("C",
    cosh(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = coshf(X);
").

:- pragma foreign_proc("C",
    tanh(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = tanhf(X);
").

:- pragma foreign_proc("C",
    atan2(Y::in, X::in) = (A::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    A = atan2f(Y, X);
").

:- pragma foreign_proc("C",
    have_fma,
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = MR_TRUE;
").

:- pragma foreign_proc("C",
    fma(X::in, Y::in, Z::in) = (F::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    F = fmaf(X, Y, Z);
").
