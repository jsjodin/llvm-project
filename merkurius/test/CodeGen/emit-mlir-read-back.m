% A module that -emit-mlir writes reads back with -x mlir and compiles to the
% same program: its table of root types comes with it, and the lowering adds
% none beside it. gc-copying.m keeps every kind of value live across
% allocations, so a collection at every allocation reads that table.
%
% RUN: merkc -fc1 -emit-mlir -O2 -triple x86_64-unknown-linux-gnu %S/gc-copying.m -o %t.mlir
% RUN: merkc -fc1 -emit-obj -O2 -x mlir %t.mlir -o %t.o
% RUN: merkc %t.o -o %t
% RUN: env MERK_GC_EVERY_ALLOC=1 %t > %t.from-mlir
% RUN: merkc -O2 %S/gc-copying.m -o %t.source
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.source > %t.from-source
% RUN: diff %t.from-source %t.from-mlir
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t > /dev/null
% REQUIRES: x86-registered-target
