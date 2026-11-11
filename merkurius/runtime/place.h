//===-- runtime/place.h - The kernel of a GPU's processes -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The table through which the host starts GPU processes in the kernel of
// their module's image, merk_place (docs/INPROGRESS.md, GPU processes that
// run loops, one kernel for a GPU's processes), shared by the host's runtime
// (gpu-processes.cpp) and the device's (device.cpp).
//
// The host launches the kernel cooperatively, so that all its teams are
// resident at once, with a slot in the table for each of its waves. The
// table is in host memory, which the device reads: a running device may
// hold stale lines of its own memory that the host writes, which its fences
// do not invalidate. To start a process, the host writes its kind, arena and
// argument words into an idle slot and then sets the slot running; lane 0
// of the slot's wave, which polls it, starts the process, and when the
// process ends sets the slot idle and tells the host (rpcProcessEnded). Once
// the table's end is set, idle waves return, and the kernel ends with them.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_PLACE_H_
#define MERKURIUS_RUNTIME_PLACE_H_

#include <stdint.h>

namespace merkurius::runtime {

// A slot's state.
constexpr uint64_t kPlaceIdle{0};
constexpr uint64_t kPlaceRunning{1};

// The threads of each of the kernel's teams.
constexpr uint32_t kPlaceTeamThreads{256};

// A wave's slot: its state, and the process's kind, by its number in the
// module (merk_process_dispatch), the address of its arena, in the device's
// memory, and the address of its argument words, in host memory.
struct PlaceSlot {
  uint64_t state;
  uint64_t kind;
  uint64_t arena;
  uint64_t words;
};

// The table: whether the kernel is to end, how many slots follow it, one
// for each of the kernel's waves, by their order in the launch, and how many
// of the kernel's teams have started, which the host waits for, as a team
// that is not resident at once with the others would never start.
struct PlaceTable {
  uint64_t end;
  uint64_t slots;
  uint64_t started;
};

inline PlaceSlot *PlaceSlots(PlaceTable *table) {
  return reinterpret_cast<PlaceSlot *>(table + 1);
}

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_PLACE_H_
