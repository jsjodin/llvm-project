% DESIGN: Sized floating point types
% A float literal's suffix is f64, f32, f16 or bf16; the lexer rejects any
% other f and digits after one, as it rejects an integer's unknown suffix.
% ORACLE: none
%
% RUN: not merkc -fc1 -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
:- module reject_float_suffix.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float.

main(!IO) :-
    io.write_float(1.0f8, !IO).
% CHECK-DAG: reject-float-suffix.m:[[@LINE-1]]:23: error: invalid float suffix
