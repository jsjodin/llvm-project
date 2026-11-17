% An element of an array of records that array.init or array.resize fills
% with, when it is all zeros, comes from a constant that every such call
% shares, not from a stack slot that each call zeroes (Lowering.cpp,
% UseZeroRecords). Lowering cannot know that empty_point is constant, as
% only inlining shows it. One that is not all zeros keeps its slot. On an
% AMD GPU the slots did not share their stack either, so a GPU process
% that made arrays of particles in seven places took 1088 bytes a lane for
% them, more than GPU processes have (2026-11-17, the Monte Carlo
% transport's credits).
%
% RUN: merkc -fc1 -O2 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=IR
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t
% RUN: %t | FileCheck %s --match-full-lines --strict-whitespace
%
% IR: @merk.zero_record.24 = private unnamed_addr constant [24 x i8] zeroinitializer
% IR-LABEL: define {{.*}}@zero_record_elements.zeros_2_0(
% IR-NOT: alloca
% IR: call ptr addrspace(1) @merk_array_new_bytes(i64 %{{[0-9]+}}, ptr nonnull @merk.zero_record.24, i64 24)
% IR-NOT: alloca
% IR: call ptr addrspace(1) @merk_array_resize_bytes(ptr addrspace(1) %{{[0-9]+}}, i64 %{{[0-9]+}}, ptr nonnull @merk.zero_record.24, i64 24)
% IR-LABEL: define {{.*}}@zero_record_elements.ones_2_0(
% IR: alloca { double, double, i64 }
% IR: call ptr addrspace(1) @merk_array_new_bytes(i64 %{{[0-9]+}}, ptr nonnull %{{[0-9]+}}, i64 24)
%
% CHECK:[point(1.5, 2.5, 7), point(0.0, 0.0, 0), point(0.0, 0.0, 0), point(0.0, 0.0, 0), point(0.0, 0.0, 0)]
% CHECK-NEXT:[point(1.0, 0.0, 1), point(1.0, 0.0, 1), point(1.0, 0.0, 1)]
% CHECK-NOT:{{.}}

:- module zero_record_elements.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list.

:- type point
    --->    point(x :: float, y :: float, tag :: int).

:- func empty_point = point.

empty_point = point(0.0, 0.0, 0).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    N = list.length(Args) + 3,
    zeros(N, Zeros),
    ones(N, Ones),
    io.write_line(array.to_list(Zeros) `with_type` list(point), !IO),
    io.write_line(array.to_list(Ones) `with_type` list(point), !IO).

    % Filled with an empty point, and grown with more.
:- pred zeros(int::in, array(point)::array_uo) is det.
:- pragma no_inline(zeros/2).

zeros(N, A) :-
    A0 = array.init(N, empty_point),
    array.resize(N + 2, empty_point, A0, A1),
    array.set(0, point(1.5, 2.5, 7), A1, A).

    % Filled with a point that is not all zeros.
:- pred ones(int::in, array(point)::array_uo) is det.
:- pragma no_inline(ones/2).

ones(N, A) :-
    A = array.init(N, point(1.0, 0.0, 1)).
