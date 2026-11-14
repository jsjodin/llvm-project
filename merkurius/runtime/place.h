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
// the device's memory, which the process's wave sets. And the process's
// incarnation, which the process's wave stores in the slot's queues as it
// starts (ProcessQueue, PeerQueue): a number the host gives each GPU process
// it starts, counted across the program, so that it grows in a slot from
// process to process, and in a peer queue from kernel to kernel.
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
// `bulkBytes` is the size from which a process's worker teams fill and copy
// its memory for it (device.cpp, Bulk), which the host chooses for the GPU.
//
// For direct sends between the GPUs of a node (docs/INPROGRESS.md, Direct
// sends between the GPUs of a node), with MERK_GPU_PEER=1: `peerQueues`, the
// address of the slots' peer queues, in the GPU's fine-grained memory
// (PeerQueue); `sendPool` and `sendPoolBytes`, the GPU's send pool, in its
// ordinary memory, from which its processes' copies of messages to other
// GPUs come; and `returns`, the GPU's return ring, in its fine-grained memory
// (ReturnRing). All are the GPU's, not the kernel's, and outlive it (see
// PeerQueue). Without the peer path they are 0.
struct PlaceTable {
  uint64_t end;
  uint64_t slots;
  uint64_t started;
  uint64_t queues;
  uint64_t check;
  int64_t blocks;
  uint64_t bulkBytes;
  uint64_t peerQueues;
  uint64_t sendPool;
  uint64_t sendPoolBytes;
  uint64_t returns;
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

// The queue through which the process of a slot receives from processes on
// the node's other GPUs (docs/INPROGRESS.md, Direct sends between the GPUs
// of a node, points 2 to 4). It is in the receiving GPU's fine-grained
// memory, as a word that a peer writes in ordinary memory is never seen by a
// poller there, and its atomics are at system scope. Its ring and `senders`
// work as ProcessQueue's, but `state` is the incarnation of the slot's last
// process shifted left by one, with 1 while it runs: one word, so that a
// sender never sees a new incarnation with the old process's openness, nor
// the old one open. A full slot makes the sender wait, as it has no spill
// list on another GPU; the closing receiver
// takes what is queued while it waits for the senders in flight. A slot
// holds the message: a word itself; for an array, the address of the
// sender's copy, in the sending GPU's send pool, which the receiver copies
// from; for a term, the term as the sender copied it there, which the
// receiver copies by its type_info. `copy` is the sender's block in its
// pool that holds the copy, or 0 for a word or a term with no cells, and
// `returnTo` the sending GPU's return ring, onto which the receiver puts
// `copy` when it is done with it, or closes with it still queued. The
// receiver knows the messages' kind, as a mailbox's are of one.
//
// The memory is NEVER FREED BEFORE THE PROGRAM ENDS. A sender on another
// GPU keeps its route, the queue's address, for its life, and may send after
// the receiving process, and its kernel, have ended; the queue must still be
// there to say it is closed. So a kernel's peer queues are a block of the
// GPU's, which goes back on the GPU's list as the kernel ends, closed, and
// the GPU's next kernel with no more slots takes it again (gpu-processes.cpp,
// PeerBlock). A GPU keeps as many blocks as it ever had kernels at once,
// each of the largest number of slots it took. The incarnation grows across
// kernels, so a stale sender finds a later one and drops its message, as a
// send to an ended process does (DirectSend); the device must never set a
// peer queue's incarnation back, nor clear a reused one. The host makes a
// new block empty: state 0, no senders, each slot's sequence its index.
constexpr uint64_t kPeerQueueSlots{64};

struct PeerQueueSlot {
  uint64_t sequence;
  uint64_t message;
  uint64_t copy;
  uint64_t returnTo;
};

struct PeerQueue {
  alignas(64) uint64_t state;
  uint64_t senders;
  alignas(64) uint64_t tail;
  alignas(64) uint64_t head;
  alignas(64) PeerQueueSlot slots[kPeerQueueSlots];
};

// A GPU's send pool, in its ordinary memory, for the program's life: the
// blocks of its processes' copies of messages to other GPUs, which receivers
// there copy from. A buddy allocator on the device keeps it
// (device/device.cpp, Send pool): blocks of kSendPoolBlock bytes and up, by
// powers of two, each with a header of two words. The pool starts with a
// SendPool, then a byte for each kSendPoolBlock bytes of the pool, which say
// which blocks are free; the host clears both (SendPoolClearBytes), and the
// first sender to allocate makes the rest (`ready`). Only the pool's GPU
// allocates and frees, under `lock`, at the device's scope.
constexpr uint64_t kSendPoolBlock{1024};
constexpr uint64_t kSendPoolOrders{48};

struct SendPool {
  uint64_t lock;
  uint64_t ready;
  uint64_t data;
  uint64_t dataBytes;
  uint64_t free[kSendPoolOrders];
};

constexpr uint64_t SendPoolClearBytes(uint64_t poolBytes) {
  return sizeof(SendPool) + poolBytes / kSendPoolBlock;
}

// A GPU's return ring, in its fine-grained memory, for the program's life:
// receivers on other GPUs put on it the blocks of the GPU's send pool that
// they are done with, and the GPU's senders take them off and free them. A
// ring of slots as a queue's, many writers and the GPU's senders as its
// reader, one at a time under the pool's lock. It has a slot for each
// block the pool could hold, so a writer never finds it full
// (ReturnRingSlots), and `slots` slots follow it.
struct ReturnRingSlot {
  uint64_t sequence;
  uint64_t address;
};

struct ReturnRing {
  alignas(64) uint64_t tail;
  alignas(64) uint64_t head;
  alignas(64) uint64_t slots;
};

constexpr uint64_t ReturnRingSlots(uint64_t poolBytes) {
  return poolBytes / kSendPoolBlock;
}

inline ReturnRingSlot *RingSlots(ReturnRing *ring) {
  return reinterpret_cast<ReturnRingSlot *>(ring + 1);
}

static_assert(sizeof(PeerQueue) % 64 == 0 && sizeof(ReturnRing) % 64 == 0,
    "the host makes their images in memory aligned as cache lines");

inline PlaceSlot *PlaceSlots(PlaceTable *table) {
  return reinterpret_cast<PlaceSlot *>(table + 1);
}

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_PLACE_H_
