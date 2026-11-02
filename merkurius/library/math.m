% A stub of the standard library module math, with its functions
% implemented with C's math library but solve_quadratic. The declarations
% are those of the library reference manual (22.01); see README.md. A
% function with a domain restriction checks it, as mmc's programs do, and
% throws domain_error("math.Name") if it does not hold.

:- module math.
:- interface.

:- func pi = float.
:- func e = float.
:- func ceiling(float) = float.
:- func floor(float) = float.
:- func round(float) = float.
:- func truncate(float) = float.
:- func sqrt(float) = float.
:- func unchecked_sqrt(float) = float.
:- type quadratic_roots
    --->    no_roots
    ;       one_root(float)
    ;       two_roots(float, float).
:- func solve_quadratic(float, float, float) = quadratic_roots.
:- func pow(float, float) = float.
:- func unchecked_pow(float, float) = float.
:- func exp(float) = float.
:- func ln(float) = float.
:- func unchecked_ln(float) = float.
:- func log10(float) = float.
:- func unchecked_log10(float) = float.
:- func log2(float) = float.
:- func unchecked_log2(float) = float.
:- func log(float, float) = float.
:- func unchecked_log(float, float) = float.
:- func sin(float) = float.
:- func cos(float) = float.
:- func tan(float) = float.
:- func asin(float) = float.
:- func unchecked_asin(float) = float.
:- func acos(float) = float.
:- func unchecked_acos(float) = float.
:- func atan(float) = float.
:- func atan2(float, float) = float.
:- func sinh(float) = float.
:- func cosh(float) = float.
:- func tanh(float) = float.
:- pred have_fma is semidet.
:- func fma(float, float, float) = float.

:- implementation.

:- pragma foreign_decl("C", local, "
#include <float.h>
#include <math.h>
#include <stdio.h>

void merk_uncaught_exception(const char *exception)
    __attribute__((noreturn));

/* Until exceptions are implemented, nothing catches one. */
static void merk_domain_error(const char *what)
{
    char buffer[256];
    snprintf(buffer, sizeof buffer, \"domain_error(\\\"%s\\\")\", what);
    merk_uncaught_exception(buffer);
}
").

:- pragma foreign_proc("C",
    pi = (P::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    P = 3.14159265358979323846;
").

:- pragma foreign_proc("C",
    e = (E::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    E = 2.7182818284590452354;
").

% round(X) is floor(X + 0.5): a fraction of 0.5 rounds up.
:- pragma foreign_proc("C",
    ceiling(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = ceil(X);
").

:- pragma foreign_proc("C",
    floor(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = floor(X);
").

:- pragma foreign_proc("C",
    round(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = floor(X + 0.5);
").

:- pragma foreign_proc("C",
    truncate(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = trunc(X);
").

:- pragma foreign_proc("C",
    sqrt(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    if (X < 0.0) {
        merk_domain_error(\"math.sqrt\");
    }
    Y = sqrt(X);
").

:- pragma foreign_proc("C",
    unchecked_sqrt(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = sqrt(X);
").

:- pragma foreign_proc("C",
    pow(X::in, P::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    if (X < 0.0 || (X == 0.0 && P <= 0.0)) {
        merk_domain_error(\"math.pow\");
    }
    Y = pow(X, P);
").

:- pragma foreign_proc("C",
    unchecked_pow(X::in, P::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = pow(X, P);
").

:- pragma foreign_proc("C",
    ln(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    if (X <= 0.0) {
        merk_domain_error(\"math.ln\");
    }
    Y = log(X);
").

:- pragma foreign_proc("C",
    unchecked_ln(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = log(X);
").

:- pragma foreign_proc("C",
    log10(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    if (X <= 0.0) {
        merk_domain_error(\"math.log10\");
    }
    Y = log10(X);
").

:- pragma foreign_proc("C",
    unchecked_log10(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = log10(X);
").

:- pragma foreign_proc("C",
    log2(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    if (X <= 0.0) {
        merk_domain_error(\"math.log2\");
    }
    Y = log2(X);
").

:- pragma foreign_proc("C",
    unchecked_log2(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = log2(X);
").

:- pragma foreign_proc("C",
    log(B::in, X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    if (X <= 0.0 || B <= 0.0 || B == 1.0) {
        merk_domain_error(\"math.log\");
    }
    Y = log(X) / log(B);
").

:- pragma foreign_proc("C",
    unchecked_log(B::in, X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = log(X) / log(B);
").

:- pragma foreign_proc("C",
    asin(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    if (X < -1.0 || X > 1.0) {
        merk_domain_error(\"math.asin\");
    }
    Y = asin(X);
").

:- pragma foreign_proc("C",
    unchecked_asin(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = asin(X);
").

:- pragma foreign_proc("C",
    acos(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    if (X < -1.0 || X > 1.0) {
        merk_domain_error(\"math.acos\");
    }
    Y = acos(X);
").

:- pragma foreign_proc("C",
    unchecked_acos(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = acos(X);
").

:- pragma foreign_proc("C",
    exp(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = exp(X);
").

:- pragma foreign_proc("C",
    sin(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = sin(X);
").

:- pragma foreign_proc("C",
    cos(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = cos(X);
").

:- pragma foreign_proc("C",
    tan(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = tan(X);
").

:- pragma foreign_proc("C",
    atan(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = atan(X);
").

:- pragma foreign_proc("C",
    sinh(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = sinh(X);
").

:- pragma foreign_proc("C",
    cosh(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = cosh(X);
").

:- pragma foreign_proc("C",
    tanh(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Y = tanh(X);
").

:- pragma foreign_proc("C",
    atan2(Y::in, X::in) = (A::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    A = atan2(Y, X);
").

:- pragma foreign_proc("C",
    have_fma,
    [will_not_call_mercury, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = MR_TRUE;
").

:- pragma foreign_proc("C",
    fma(X::in, Y::in, Z::in) = (F::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    F = fma(X, Y, Z);
").
