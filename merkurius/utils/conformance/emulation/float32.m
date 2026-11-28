% float32 for mmc, which has none: the oracle's stand-in for Merkurius'
% library/float32.m, for the float32 instances of templated conformance
% tests (utils/conformance/instantiate.py). A float32 is a float that
% holds a value of C's float, and each operation computes in C's float, so
% it rounds as merkc's float32 does. A literal 1.5f32 is lit("1.5") here,
% rounded once from its digits. io.write writes a float32 as
% float32_value(1.5), which oracle.py rewrites into 1.5f32, as merkc's
% programs write it. io.write_float32 and string.float32_to_string are
% write_float32 and float32_to_string here.

:- module float32.
:- interface.
:- import_module io.

:- type float32
    --->    float32_value(float).

:- func lit(string) = float32.

:- func (float32::in) + (float32::in) = (float32::out) is det.
:- func (float32::in) - (float32::in) = (float32::out) is det.
:- func (float32::in) * (float32::in) = (float32::out) is det.
:- func (float32::in) / (float32::in) = (float32::out) is det.
:- func unchecked_quotient(float32::in, float32::in) = (float32::out) is det.
:- func + (float32::in) = (float32::out) is det.
:- func - (float32::in) = (float32::out) is det.
:- pred (float32::in) < (float32::in) is semidet.
:- pred (float32::in) =< (float32::in) is semidet.
:- pred (float32::in) >= (float32::in) is semidet.
:- pred (float32::in) > (float32::in) is semidet.
:- func from_float(float) = float32.
:- func to_float(float32) = float.
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

:- pred write_float32(float32::in, io::di, io::uo) is det.
:- func float32_to_string(float32) = string.

:- implementation.
:- import_module exception, float, int.

:- pragma foreign_decl("C", local, "
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* As runtime/write.cpp's merk_format_float32: the fewest of 6 to 9
   digits that read back as the same float. */
static void emulated_format_float32(float x, char *b)
{
    if (isnan(x)) {
        strcpy(b, \"nan\");
    } else if (isinf(x)) {
        strcpy(b, x < 0 ? \"-infinity\" : \"infinity\");
    } else {
        int p;
        for (p = 6; p <= 9; ++p) {
            snprintf(b, 64, \"%.*g\", p, (double) x);
            if (strtof(b, NULL) == x) {
                break;
            }
        }
        if (strpbrk(b, \".e\") == NULL) {
            strcat(b, \".0\");
        }
    }
}

static MR_Integer emulated_to_int(float x)
{
    return x >= -0x1p63f && x < 0x1p63f ? (MR_Integer) x : INTPTR_MIN;
}
").

:- func f(float32) = float.
f(float32_value(X)) = X.

% The C parts take and return floats, as mmc passes a float32 as a word.

    % The float rounded to C's float, as a float.
:- func round(float) = float.
:- pragma foreign_proc("C",
    round(X::in) = (R::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    R = (double) (float) X;
").

lit(S) = float32_value(lit_value(S)).
:- func lit_value(string) = float.
:- pragma foreign_proc("C",
    lit_value(S::in) = (R::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    R = (double) strtof(S, NULL);
").

X + Y = float32_value(round(f(X) + f(Y))).
X - Y = float32_value(round(f(X) - f(Y))).
X * Y = float32_value(round(f(X) * f(Y))).
X / Y = Q :-
    ( if f(Y) = 0.0 then
        throw(domain_error("float32.'/': division by zero"))
    else
        Q = unchecked_quotient(X, Y)
    ).
unchecked_quotient(X, Y) =
    float32_value(round(float.unchecked_quotient(f(X), f(Y)))).
+ X = X.
% As merkc's: 0.0 - X, so -(0.0) is 0.0.
- X = float32_value(round(0.0 - f(X))).
X < Y :- f(X) < f(Y).
X =< Y :- f(X) =< f(Y).
X >= Y :- f(X) >= f(Y).
X > Y :- f(X) > f(Y).

from_float(X) = float32_value(round(X)).
to_float(X) = f(X).
cast_from_int(N) = float32_value(int_value(N)).
:- func int_value(int) = float.
:- pragma foreign_proc("C",
    int_value(N::in) = (R::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    R = (double) (float) N;
").

ceiling_to_int(F) = to_int(0, f(F)).
floor_to_int(F) = to_int(1, f(F)).
round_to_int(F) = to_int(2, f(F)).
truncate_to_int(F) = to_int(3, f(F)).
    % As library/float32.m's conversions: in C's float, and min_int out of
    % int's range.
:- func to_int(int, float) = int.
:- pragma foreign_proc("C",
    to_int(How::in, F::in) = (N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    float x = (float) F;
    switch (How) {
    case 0: x = ceilf(x); break;
    case 1: x = floorf(x); break;
    case 2: x = floorf(x + 0.5f); break;
    default: break;
    }
    N = emulated_to_int(x);
").

abs(X) = float32_value(float.abs(f(X))).
max(X, Y) = ( if X >= Y then X else Y ).
min(X, Y) = ( if X =< Y then X else Y ).

pow(X, N) = P :-
    ( if f(X) = 0.0, N < 0 then
        throw(domain_error("float32.pow: zero base"))
    else
        P0 = squaring(X, int.abs(N), float32_value(1.0)),
        P = ( if N < 0 then unchecked_quotient(float32_value(1.0), P0)
            else P0 )
    ).

:- func squaring(float32, int, float32) = float32.
squaring(X, N, P) =
    ( if N > 0 then
        squaring(X * X, N >> 1, ( if N /\ 1 = 1 then P * X else P ))
    else
        P
    ).

is_infinite(X) :- float.is_infinite(f(X)).
is_inf(X) :- float.is_infinite(f(X)).
is_nan(X) :- float.is_nan(f(X)).
is_nan_or_infinite(X) :- float.is_nan_or_infinite(f(X)).
is_nan_or_inf(X) :- float.is_nan_or_infinite(f(X)).
is_finite(X) :- float.is_finite(f(X)).
is_zero(X) :- f(X) = 0.0.

max = float32_value(constant(0)).
min = float32_value(constant(1)).
infinity = float32_value(float.infinity).
epsilon = float32_value(constant(2)).
:- func constant(int) = float.
:- pragma foreign_proc("C",
    constant(Which::in) = (F::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    F = Which == 0 ? FLT_MAX : Which == 1 ? FLT_MIN : FLT_EPSILON;
").
radix = 2.
mantissa_digits = 24.
min_exponent = -125.
max_exponent = 128.

write_float32(X, !IO) :-
    io.write_string(float32_to_string(X), !IO).

float32_to_string(X) = float_text(f(X)).
:- func float_text(float) = string.

:- pragma foreign_proc("C",
    float_text(F::in) = (S::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    char buffer[64];
    emulated_format_float32((float) F, buffer);
    MR_make_aligned_string_copy(S, buffer);
").
