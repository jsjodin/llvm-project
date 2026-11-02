% Constants of the primitive types: integer literals in each base, with
% `_` and suffixes, and 0'c; floats; characters, which are their code
% points; and strings, with their escapes replaced.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM

% MERK-LABEL: merk.proc @constants.main_2_0()
% MERK-NEXT: merk.constant 42 : i64
% MERK-NEXT: merk.constant 1.000500e+03 : f64
% MERK-NEXT: merk.constant 233 : i32
% MERK-NEXT: merk.constant "a\09b\0A\22" : !merk.string
% MERK-NEXT: merk.constant -5 : i8
% MERK-NEXT: merk.constant 7 : i64
% MERK-NEXT: merk.call @constants.p_6_0
% MERK-NEXT: merk.constant 97 : i64
% MERK-NEXT: merk.constant 2.000000e+03 : f64
% MERK-NEXT: merk.constant 128512 : i32
% MERK-NEXT: merk.constant "" : !merk.string
% MERK-NEXT: merk.constant 100 : i8
% MERK-NEXT: merk.constant 5 : i64
% MERK-NEXT: merk.call @constants.p_6_0
% MERK-NEXT: merk.constant 1000000 : i64
% MERK-NEXT: merk.constant 255 : i64
% MERK-NEXT: merk.constant 8 : i64
% MERK-NEXT: merk.constant -1 : i64
% MERK-NEXT: merk.call @constants.q_4_0
% MERK-NEXT: merk.constant "a\09b\0A\22" : !merk.string
% MERK-NEXT: merk.call @constants.r_1_0
% MERK-NEXT: merk.succeed

% Equal strings share one global.
% LLVM-DAG: @merk.str.0 = private unnamed_addr constant [6 x i8] c"a\09b\0A\22\00"
% LLVM-DAG: @merk.str.1 = private unnamed_addr constant [1 x i8] zeroinitializer
% LLVM-LABEL: define tailcc void @constants.main_2_0()
% LLVM-NEXT: call tailcc void @constants.p_6_0(i64 42, double 1.000500e+03, i32 233, ptr @merk.str.0, i8 -5, i64 7)
% LLVM-NEXT: call tailcc void @constants.p_6_0(i64 97, double 2.000000e+03, i32 128512, ptr @merk.str.1, i8 100, i64 5)
% LLVM-NEXT: call tailcc void @constants.q_4_0(i64 1000000, i64 255, i64 8, i64 -1)
% LLVM-NEXT: call tailcc void @constants.r_1_0(ptr @merk.str.0)

:- module constants.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char.

:- pred p(int::in, float::in, char::in, string::in, int8::in, uint::in) is det.
p(_, _, _, _, _, _).

:- pred q(int::in, int::in, int::in, int::in) is det.
q(_, _, _, _).

:- pred r(string::in) is det.
r(_).

main(!IO) :-
    p(0x2a, 1_000.5, 'é', "a\tb\n\"", -5i8, 7u),
    p(0'a, 2.0e3, '\x1F600\', "", 100i8, 0b101u),
    q(1_000_000, 0xff, 0o10, -1),
    r("a\tb\n\"").
