//===-- runtime/process.cpp - Merkurius processes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "process.h"

namespace merkurius::runtime {
namespace {
// Constant-initialized, as merk_heap is.
thread_local Process *current{nullptr};
Process firstProcess{{nullptr, nullptr}, 1};
} // namespace

Process *CurrentProcess() { return current; }

void RunProcess(Process *next) {
  if (current) {
    current->heap = merk_heap;
  }
  merk_heap = next ? next->heap : MerkHeap{nullptr, nullptr};
  current = next;
}

} // namespace merkurius::runtime

extern "C" void merk_process_start(void) {
  merkurius::runtime::RunProcess(&merkurius::runtime::firstProcess);
}
