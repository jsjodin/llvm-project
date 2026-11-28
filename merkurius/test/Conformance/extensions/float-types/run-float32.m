% DESIGN: Sized floating point types
% float32 is IEEE single precision, C's float. A literal with the suffix
% f32 rounds once to it; its operators round each result to it; from_float
% rounds a float to the nearest float32 and to_float is exact. A float32
% compares as a float does: 0.0f32 = -0.0f32. io.write writes one with its
% suffix, with the fewest of 6 to 9 digits that read back as the same
% float32, as write_float32 and float32_to_string give it without the
% suffix; an infinity has none. Lists, tuples, records' fields and arrays
% hold float32s, an array packed 4 bytes to an element. math_float32 has
% math's functions under math's names, so sqrt of a float32 is its own.
% A literal too large for float32 is an infinity, as one too large for
% float is. A float32 constant and a float constant of the same bits are
% different values. f64 is float's suffix, as float64 is float.
% TRANSLATION: Inputs/float32_translation.c
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1.6
% OUTPUT-NEXT:0.10000000149011612
% OUTPUT-NEXT:0.1
% OUTPUT-NEXT:0.5
% OUTPUT-NEXT:0.33333334
% OUTPUT-NEXT:[1.5f32, 0.1f32, -2.5f32, 1e+10f32, infinity]
% OUTPUT-NEXT:[3.0f32, 0.2f32]
% OUTPUT-NEXT:["half", "tenth", "other"]
% OUTPUT-NEXT:{0.5, 0.5f32}
% OUTPUT-NEXT:point(1.0f32, 7, 2e+10f32)
% OUTPUT-NEXT:2e+10f32
% OUTPUT-NEXT:array([0.25f32, 0.25f32, 7.5f32, 0.25f32])
% OUTPUT-NEXT:8.25f32
% OUTPUT-NEXT:array([point(1.0f32, 1, 2.0f32), point(3.0f32, 2, 4.0f32)])
% OUTPUT-NEXT:gt
% OUTPUT-NEXT:zeros equal
% OUTPUT-NEXT:'<'
% OUTPUT-NEXT:2
% OUTPUT-NEXT:-2
% OUTPUT-NEXT:16777216.0
% OUTPUT-NEXT:3.4028235e+38
% OUTPUT-NEXT:1.1754944e-38
% OUTPUT-NEXT:1.1920929e-07
% OUTPUT-NEXT:infinity
% OUTPUT-NEXT:3.375
% OUTPUT-NEXT:1.4142135
% OUTPUT-NEXT:1.4142135623730951
% OUTPUT-NEXT:3.1415927
% OUTPUT-NEXT:2.7182817
% OUTPUT-NEXT:3.1415927
% OUTPUT-NEXT:3.0
% OUTPUT-NEXT:{1e+10f32, 0.0025f32, 1.0, 300.0f32}
% OUTPUT-NOT:{{.}}
:- module run_float32.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, float, float32, int, list, math, math_float32,
    string.

:- type point
    --->    point(x :: float32, n :: int, y :: float32).

:- func scale(float32, float32) = float32.
scale(K, X) = K * X.

:- func name(float32) = string.
name(X) = N :-
    ( if X = 0.5f32 then
        N = "half"
    else if X = 0.1f32 then
        N = "tenth"
    else
        N = "other"
    ).

:- func sum(array(float32)) = float32.
sum(A) = array.foldl(func(X, S) = X + S, A, 0.0f32).

main(!IO) :-
    A = 1.5f32,
    B = 0.1f32,
    io.write_float32(A + B, !IO), io.nl(!IO),
    io.write_float(float32.to_float(B), !IO), io.nl(!IO),
    io.write_float32(float32.from_float(0.1), !IO), io.nl(!IO),
    io.write_string(string.float32_to_string(A / 3.0f32), !IO), io.nl(!IO),
    io.write_float32(1.0f32 / 3.0f32, !IO), io.nl(!IO),
    io.write([A, B, -2.5f32, 1.0e10f32, 1.0e39f32], !IO), io.nl(!IO),
    io.write(list.map(scale(2.0f32), [A, B]), !IO), io.nl(!IO),
    io.write(list.map(name, [0.5f32, 0.1f32, 2.0f32]), !IO), io.nl(!IO),
    io.write({0.5, 0.5f32}, !IO), io.nl(!IO),
    P = point(1.0f32, 7, 2.0e10f32),
    io.write(P, !IO), io.nl(!IO),
    io.write(P ^ y - P ^ x, !IO), io.nl(!IO),
    Arr0 = array.init(4, 0.25f32),
    array.set(2, 7.5f32, Arr0, Arr),
    io.write(Arr, !IO), io.nl(!IO),
    io.write(sum(Arr), !IO), io.nl(!IO),
    io.write(array.from_list([point(1.0f32, 1, 2.0f32), point(3.0f32, 2, 4.0f32)]) : array(point), !IO), io.nl(!IO),
    ( if A > B then io.write_string("gt\n", !IO) else io.write_string("le\n", !IO) ),
    ( if 0.0f32 = -0.0f32 then io.write_string("zeros equal\n", !IO) else io.write_string("zeros differ\n", !IO) ),
    compare(R, B, A), io.write(R, !IO), io.nl(!IO),
    io.write_int(float32.truncate_to_int(2.75f32), !IO), io.nl(!IO),
    io.write_int(float32.round_to_int(-2.5f32), !IO), io.nl(!IO),
    io.write_float32(float32.cast_from_int(16777217), !IO), io.nl(!IO),
    io.write_float32(float32.max, !IO), io.nl(!IO),
    io.write_float32(float32.min, !IO), io.nl(!IO),
    io.write_float32(float32.epsilon, !IO), io.nl(!IO),
    io.write_float32(float32.from_float(1.0e300), !IO), io.nl(!IO),
    io.write_float32(float32.pow(1.5f32, 3), !IO), io.nl(!IO),
    io.write_float32(sqrt(2.0f32), !IO), io.nl(!IO),
    io.write_float(sqrt(2.0), !IO), io.nl(!IO),
    io.write_float32(math_float32.pi, !IO), io.nl(!IO),
    io.write_float32(exp(1.0f32), !IO), io.nl(!IO),
    io.write_float32(atan2(1.0f32, 1.0f32) * 4.0f32, !IO), io.nl(!IO),
    io.write_float32(log(2.0f32, 8.0f32), !IO), io.nl(!IO),
    io.write({1e+10f32, 2.5e-3f32, 1.0f64, 3.0E2f32}, !IO), io.nl(!IO).
