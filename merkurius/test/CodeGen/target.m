% The target is recorded on the module (TargetInfo): the triple, the data
% layout, in LLVM's form and DLTI's, the CPU, the optimization level and the
% address spaces of Mercury code, which are all 0 on a CPU. Read back from
% its text, a module lowers for the target it records, with no -triple.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=X86
% X86: llvm.data_layout = "e-m:e-
% X86-SAME: llvm.target_triple = "x86_64-unknown-linux-gnu"
% X86-SAME: merk.address_spaces = #merk.address_spaces<generic = 0, global = 0, shared = 0, constant = 0, private = 0>
% X86-SAME: merk.optimization_level = 0 : i32
%
% RUN: merkc -fc1 -emit-merk -triple amdgcn-amd-amdhsa -target-cpu gfx90a -O2 %s -o %t.amdgpu.mlir
% RUN: FileCheck %s --check-prefix=AMDGPU < %t.amdgpu.mlir
% AMDGPU: dlti.dl_spec = #dlti.dl_spec<
% AMDGPU-SAME: "dlti.alloca_memory_space" = 5 : ui64
% AMDGPU-SAME: llvm.target_triple = "amdgcn-amd-amdhsa"
% AMDGPU-SAME: merk.address_spaces = #merk.address_spaces<generic = 0, global = 1, shared = 3, constant = 4, private = 5>
% AMDGPU-SAME: merk.optimization_level = 2 : i32
% AMDGPU-SAME: merk.target_cpu = "gfx90a"
%
% Written again from its text, the module is the same.
% RUN: merkc -fc1 -emit-merk %t.amdgpu.mlir -o %t.again.mlir
% RUN: diff %t.amdgpu.mlir %t.again.mlir
%
% RUN: merkc -fc1 -emit-merk -triple aarch64-unknown-linux-gnu %s -o %t.aarch64.mlir
% RUN: merkc -fc1 -emit-llvm %t.aarch64.mlir | FileCheck %s --check-prefix=AARCH64
% AARCH64: target triple = "aarch64-unknown-linux-gnu"
% AARCH64: define tailcc void @target.main_2_0()
%
% Without -O, code generation for an AMD GPU is at -O2, as Clang's linker
% wrapper compiles OpenMP's device code, while the passes before it are at
% -O0; the module records that no -O was given. An -O that was given is
% respected. The code is llc's at -O2 and at -O0 for the module's IR.
% RUN: merkc -fc1 -emit-merk -triple amdgpu9.0a-amd-amdhsa %s | FileCheck %s --check-prefix=DEFAULT
% DEFAULT: merk.optimization_level = 0 : i32, merk.optimization_level_default
% RUN: merkc -fc1 -emit-llvm -triple amdgpu9.0a-amd-amdhsa %s -o %t.amdgpu.ll
% RUN: merkc -fc1 -S -triple amdgpu9.0a-amd-amdhsa %s -o - | grep -E '^\s+[sv]_' > %t.default.s
% RUN: merkc -fc1 -S -O0 -triple amdgpu9.0a-amd-amdhsa %s -o - | grep -E '^\s+[sv]_' > %t.O0.s
% RUN: llc -O2 %t.amdgpu.ll -o - | grep -E '^\s+[sv]_' > %t.llc2.s
% RUN: llc -O0 %t.amdgpu.ll -o - | grep -E '^\s+[sv]_' > %t.llc0.s
% RUN: diff %t.default.s %t.llc2.s
% RUN: diff %t.O0.s %t.llc0.s
% RUN: not diff %t.default.s %t.O0.s
%
% A module without a target cannot be lowered.
% RUN: sed -e 's/llvm.target_triple = "[^"]*", //' %t.aarch64.mlir > %t.none.mlir
% RUN: not merkc -fc1 -emit-llvm %t.none.mlir 2>&1 | FileCheck %s --check-prefix=NONE
% NONE: merkc: error: {{.*}}none.mlir: the module records no target triple
%
% An unknown target is an error.
% RUN: not merkc -fc1 -emit-merk -triple nosuch-unknown-none %s 2>&1 | FileCheck %s --check-prefix=UNKNOWN
% UNKNOWN: merkc: error: {{.*}}nosuch

:- module target.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
main(!IO).
