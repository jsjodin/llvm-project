% math_float32 for mmc, which has none: the oracle's stand-in for
% Merkurius' library/math_float32.m, for the float32 instances of templated
% conformance tests (utils/conformance/instantiate.py). Each function is
% C's single-precision one, as the library's are on the host, with the
% library's domain checks, which throw domain_error("math_float32.Name").


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
:- import_module exception, float.

:- func f(float32) = float.
f(float32_value(X)) = X.

    % C's function number Which of X and Y, in C's float. mmc passes a
    % float32 as a word, so the C part takes floats.
:- func single(int, float, float) = float.
:- pragma foreign_proc("C",
    single(Which::in, X0::in, Y0::in) = (R::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    float x = (float) X0, y = (float) Y0, r;
    switch (Which) {
    case 0: r = acosf(x); break;
    case 1: r = asinf(x); break;
    case 2: r = atanf(x); break;
    case 3: r = atan2f(x, y); break;
    case 4: r = ceilf(x); break;
    case 5: r = cosf(x); break;
    case 6: r = coshf(x); break;
    case 7: r = expf(x); break;
    case 8: r = floorf(x); break;
    case 9: r = logf(x); break;
    case 10: r = logf(y) / logf(x); break;
    case 11: r = log10f(x); break;
    case 12: r = log2f(x); break;
    case 13: r = powf(x, y); break;
    case 14: r = floorf(x + 0.5f); break;
    case 15: r = sinf(x); break;
    case 16: r = sinhf(x); break;
    case 17: r = sqrtf(x); break;
    case 18: r = tanf(x); break;
    case 19: r = tanhf(x); break;
    case 20: r = truncf(x); break;
    default: r = 0.0f; break;
    }
    R = (double) r;
").

pi = float32_value(pi_value).
e = float32_value(e_value).
:- func pi_value = float.
:- pragma foreign_proc("C", pi_value = (P::out),
    [will_not_call_mercury, promise_pure, thread_safe],
    "P = (double) 3.14159265358979323846f;").
:- func e_value = float.
:- pragma foreign_proc("C", e_value = (E::out),
    [will_not_call_mercury, promise_pure, thread_safe],
    "E = (double) 2.7182818284590452354f;").

ceiling(X) = float32_value(single(4, f(X), 0.0)).
floor(X) = float32_value(single(8, f(X), 0.0)).
round(X) = float32_value(single(14, f(X), 0.0)).
truncate(X) = float32_value(single(20, f(X), 0.0)).
sqrt(float32_value(X)) = Y :-
    ( if X < 0.0 then
        throw(domain_error("math_float32.sqrt"))
    else
        Y = float32_value(single(17, X, 0.0))
    ).
unchecked_sqrt(X) = float32_value(single(17, f(X), 0.0)).
exp(X) = float32_value(single(7, f(X), 0.0)).
ln(float32_value(X)) = Y :-
    ( if X =< 0.0 then
        throw(domain_error("math_float32.ln"))
    else
        Y = float32_value(single(9, X, 0.0))
    ).
unchecked_ln(X) = float32_value(single(9, f(X), 0.0)).
log10(float32_value(X)) = Y :-
    ( if X =< 0.0 then
        throw(domain_error("math_float32.log10"))
    else
        Y = float32_value(single(11, X, 0.0))
    ).
unchecked_log10(X) = float32_value(single(11, f(X), 0.0)).
log2(float32_value(X)) = Y :-
    ( if X =< 0.0 then
        throw(domain_error("math_float32.log2"))
    else
        Y = float32_value(single(12, X, 0.0))
    ).
unchecked_log2(X) = float32_value(single(12, f(X), 0.0)).
sin(X) = float32_value(single(15, f(X), 0.0)).
cos(X) = float32_value(single(5, f(X), 0.0)).
tan(X) = float32_value(single(18, f(X), 0.0)).
asin(float32_value(X)) = Y :-
    ( if ( X < -1.0 ; X > 1.0 ) then
        throw(domain_error("math_float32.asin"))
    else
        Y = float32_value(single(1, X, 0.0))
    ).
unchecked_asin(X) = float32_value(single(1, f(X), 0.0)).
acos(float32_value(X)) = Y :-
    ( if ( X < -1.0 ; X > 1.0 ) then
        throw(domain_error("math_float32.acos"))
    else
        Y = float32_value(single(0, X, 0.0))
    ).
unchecked_acos(X) = float32_value(single(0, f(X), 0.0)).
atan(X) = float32_value(single(2, f(X), 0.0)).
sinh(X) = float32_value(single(16, f(X), 0.0)).
cosh(X) = float32_value(single(6, f(X), 0.0)).
tanh(X) = float32_value(single(19, f(X), 0.0)).
pow(float32_value(X), float32_value(P)) = R :-
    ( if ( X < 0.0 ; X = 0.0, P =< 0.0 ) then
        throw(domain_error("math_float32.pow"))
    else
        R = float32_value(single(13, X, P))
    ).
unchecked_pow(float32_value(X), float32_value(P)) = float32_value(single(13, X, P)).
log(float32_value(B), float32_value(X)) = R :-
    ( if ( X =< 0.0 ; B =< 0.0 ; B = 1.0 ) then
        throw(domain_error("math_float32.log"))
    else
        R = float32_value(single(10, B, X))
    ).
unchecked_log(float32_value(B), float32_value(X)) = float32_value(single(10, B, X)).
atan2(float32_value(Y), float32_value(X)) = float32_value(single(3, Y, X)).
have_fma :- semidet_true.
fma(float32_value(X), float32_value(Y), float32_value(Z)) =
    float32_value(fma_value(X, Y, Z)).
:- func fma_value(float, float, float) = float.
:- pragma foreign_proc("C",
    fma_value(X::in, Y::in, Z::in) = (F::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    F = (double) fmaf((float) X, (float) Y, (float) Z);
").
