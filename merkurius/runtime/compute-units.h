//===-- runtime/compute-units.h - A GPU's units for processes ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The compute units on which a GPU's processes run (docs/INPROGRESS.md, GPU
// processes that run loops, one kernel for a GPU's processes): their kernel's
// queue is limited to those that its other queues may use, which
// LIBOMPTARGET_AMDGPU_CU_MASK may limit, as a mask with a bit for each, bit
// I % 32 of word I / 32 for unit I, as liboffload's
// olCreateQueueWithComputeUnitMask takes it. They are taken in pairs, units
// 2P and 2P + 1, as processors with work group processors enable them.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_COMPUTE_UNITS_H_
#define MERKURIUS_RUNTIME_COMPUTE_UNITS_H_

#include <cstdint>

namespace merkurius::runtime {

// A mask of up to 256 compute units.
constexpr int kUnitMaskWords{8};
struct UnitMask {
  std::uint32_t words[kUnitMaskWords];
};

int UnitCount(const UnitMask &mask);

// The pairs of units of `usable`, of `words` words, in `mask`. False if it
// has no pair, or has a unit beyond the 256th. A unit of `usable` without
// its pair is not used.
bool ProcessUnits(const std::uint32_t *usable, int words, UnitMask &mask);

// Keeps only the lowest pair of `mask`, which has one.
void FirstPair(UnitMask &mask);

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_COMPUTE_UNITS_H_
