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
// A process whose kind runs loops across teams (docs/INPROGRESS.md, GPU
// processes that run loops) also has worker teams: whole teams of the
// kernel, every slot of them idle, which the host reserves for it at the
// spawn, giving each of their slots the process's slot, the team's number
// among its workers and how many there are, before it sets the process's
// slot running. The process's wave makes its control block in its arena and
// sets each reserved slot working, with the block; the waves of a working
// team run the process's loops together, and when the process ends set their
// slots reserved again, before it tells the host, which then sets them idle.
// So the host makes slots running or reserved, and idle again, and the device
// makes them working and reserved, and idle after a process.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_PLACE_H_
#define MERKURIUS_RUNTIME_PLACE_H_

#include <stdint.h>

namespace merkurius::runtime {

// A slot's state.
constexpr uint64_t kPlaceIdle{0};
constexpr uint64_t kPlaceRunning{1};
constexpr uint64_t kPlaceReserved{2};
constexpr uint64_t kPlaceWorking{3};

// The words that a loop across a process's teams gives its body, in the
// process's control block (MLIRGenProcesses.cpp's kProcessLoopWords).
constexpr uint32_t kProcessLoopWords{32};

// The most reductions of such a loop (MLIRGenProcesses.cpp's
// kProcessLoopReductions).
constexpr uint32_t kProcessLoopReductions{8};

// The threads of each of the kernel's teams.
constexpr uint32_t kPlaceTeamThreads{256};

// A wave's slot: its state, and the process's kind, by its number in the
// module (merk_process_dispatch), the address of its arena, in the device's
// memory, and the address of its argument words, in host memory. For a
// process with worker teams, how many it has. For a slot of a worker team,
// the slot of its process, the team's number among the process's workers,
// how many there are, and the address of the process's control block, in
// the device's memory, which the process's wave sets. And how many
// processes the host has started in the slot, this one included, which the
// process's wave stores in the slot's queue as it starts (ProcessQueue).
struct PlaceSlot {
  uint64_t state;
  uint64_t kind;
  uint64_t arena;
  uint64_t words;
  uint64_t owner;
  uint64_t worker;
  uint64_t workers;
  uint64_t control;
  uint64_t incarnation;
};

// The table: whether the kernel is to end, how many slots follow it, one
// for each of the kernel's waves, by their order in the launch, and how many
// of the kernel's teams have started, which the host waits for, as a team
// that is not resident at once with the others would never start; and the
// address of the slots' queues, in the device's memory (ProcessQueue). With
// MERK_CHECK_DEVICE_MEMORY, `check` is set, and the device counts in
// `blocks` the blocks of messages between its processes that no process
// holds, and their queues' spills, which the program's end reports.
struct PlaceTable {
  uint64_t end;
  uint64_t slots;
  uint64_t started;
  uint64_t queues;
  uint64_t check;
  int64_t blocks;
};

// The queue through which the process of a slot receives from the other
// processes of the kernel, with no call of the host (docs/INPROGRESS.md,
// Messages between GPU processes on one GPU). The host allocates the
// kernel's queues before it launches the kernel, and each wave makes its
// own empty before its team counts itself started; they go with the kernel,
// so a sender never touches freed memory. A queue is the team mailboxes'
// ring (device.cpp): a sender takes a position with an atomic add on
// `tail`, and a slot's sequence says which position it holds; a sender whose
// slot is full spills to the slot's list.
//
// `incarnation` is the slot's process's (PlaceSlot), 0 before the first;
// `closed` is set by the process as it ends; `senders` counts the senders
// in flight, which the process waits for before it frees what the queue
// still holds; and `kind` is the messages' (device-ring.h's kRingOf...),
// which its senders set.
constexpr uint64_t kProcessQueueSlots{64};

struct ProcessQueueSlot {
  uint64_t sequence;
  uint64_t message;
};

struct ProcessQueue {
  alignas(64) uint64_t incarnation;
  uint64_t closed;
  uint64_t senders;
  uint64_t kind;
  alignas(64) uint64_t tail;
  alignas(64) uint64_t head;
  alignas(64) ProcessQueueSlot slots[kProcessQueueSlots];
  uint64_t spills[kProcessQueueSlots];
};

inline PlaceSlot *PlaceSlots(PlaceTable *table) {
  return reinterpret_cast<PlaceSlot *>(table + 1);
}

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_PLACE_H_
