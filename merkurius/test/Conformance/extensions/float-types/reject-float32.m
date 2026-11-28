% DESIGN: Sized floating point types
% float32 and float are different types: a literal without a suffix is a
% float, and one with f32 a float32, so neither is the other's, nor is an
% operand of the other's operator; a conversion is explicit, with
% from_float and to_float. Temporary (step 3, docs/INPROGRESS.md, Sized
% floating point types): the literals of float16 and bfloat16 are not yet
% implemented; step 3 makes them those types' values.
% ORACLE: none
%
% RUN: not merkc -fc1 -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
:- module reject_float32.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, float32.

main(!IO) :-
    W : float = 1.5f32,
% CHECK-DAG: reject-float32.m:[[@LINE-1]]:17: error: type error: `1.5f32' has type float32, but must have type float here
    V : float32 = 2.5,
% CHECK-DAG: reject-float32.m:[[@LINE-1]]:19: error: type error: `2.5' has type float, but must have type float32 here
    U = 1.0f32 + 2.0,
% CHECK-DAG: reject-float32.m:[[@LINE-1]]:18: error: type error: `2.0' has type float, but must have type float32 here
    H = 2.0f16,
% CHECK-DAG: reject-float32.m:[[@LINE-1]]:9: error: not yet implemented: the type float16 of the literal 2.0f16
    B = 3.0bf16,
% CHECK-DAG: reject-float32.m:[[@LINE-1]]:9: error: not yet implemented: the type bfloat16 of the literal 3.0bf16
    io.write_float(W, !IO),
    io.write_float32(V, !IO),
    io.write_float32(U, !IO),
    io.write_float(H, !IO),
    io.write_float(B, !IO).
