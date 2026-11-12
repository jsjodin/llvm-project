//===-- runtime/device-ring.h - A mailbox's ring for the device -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The ring through which a device receives from a mailbox, in host-pinned
// memory, which the device reads directly (docs/INPROGRESS.md, The device's
// side of the first slice, step 7). The host's list stays the mailbox's
// queue; the ring shows the device its oldest messages. For a word
// message a slot is the message, and for an array its header, which has its
// size and its elements' width (array.h). Any other
// message the device asks the host to copy where it reads it (rpcTakeTerm),
// and its slot says only that it is there. The device moves `head` when it
// has taken a message, and the host then drops it from its list.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_DEVICE_RING_H_
#define MERKURIUS_RUNTIME_DEVICE_RING_H_

#include <stdint.h>

namespace merkurius::runtime {

constexpr uint64_t kRingSlots{64};

// What a ring's mailbox holds: one-word messages, arrays of numbers, whose
// bytes are copied, or other values, which are copied by their type_info
// (copy.h).
constexpr uint64_t kRingOfWords{0};
constexpr uint64_t kRingOfArrays{1};
constexpr uint64_t kRingOfTerms{2};

// The size of a chunk of an arena that the device asks the host for, unless
// it needs a larger one for a single cell. The host keeps a few chunks of
// this size that regions have finished with, for the next to ask.
constexpr uint64_t kChunkBytes{uint64_t{1} << 20};

// The device's table of the rings it has, by the mailbox's slot, after its
// arena's current chunk and whether a wave is asking for one: the words
// that the host clears when no region is running.
constexpr uint64_t kBumpWords{2};
constexpr uint64_t kRingCacheSlots{64};
constexpr uint64_t kArenaClearWords{kBumpWords + kRingCacheSlots};

// How many regions may run at once, all told. Each running region has a
// slot, which the host gives it as it starts and takes back as it ends. A
// region's image has, for each slot, the current chunk and its flag, and
// the table of the region's teams (docs/INPROGRESS.md, Processes as teams,
// step 2): the cells a region makes are bumped from chunks of its own,
// which the host takes back when the region ends. A region that starts
// while all are taken waits for one. A slot also has the table of the rings
// the region has attached, which go with it.
//
// The host gives a slot with a generation, new each time, as the slot's
// number in the low 32 bits of the region's word and the generation in the
// high: the region's device clears the slot's words as the region starts if
// they are of another generation, which a copy from the host would take
// tens of microseconds for, every region. The generation is never 0, which
// a slot starts with, or kRegionSlotClearing.
constexpr uint64_t kRegionSlots{256};
constexpr uint64_t kRegionRingSlots{16};
constexpr uint64_t kRegionSlotWords{kBumpWords + 2 + kRegionRingSlots};
constexpr uint64_t kRegionSlotClearing{~uint64_t{0}};
constexpr uint64_t RegionSlotNumber(uint64_t word) { return word & 0xffffffff; }
constexpr uint64_t RegionSlotGeneration(uint64_t word) { return word >> 32; }
// The slot of a chunk that is not a region's.
constexpr uint64_t kNoRegionSlot{~uint64_t{0}};

// The place in a team mailbox's handle on a device, whose slot is the team.
// Temporary (GPU processes, docs/INPROGRESS.md): a GPU becomes a place of its
// node, with a number of its own, when the host can send to a team.
constexpr uint64_t kTeamsPlace{0xffff};

struct DeviceRing {
  // Written by the device, in a cache line of its own.
  alignas(64) uint64_t head;
  // Written by the host: the next slot, and whether its list holds messages
  // that did not fit, for which the device must ask (rpcDrain).
  alignas(64) uint64_t tail;
  uint64_t spilled;
  alignas(64) uint64_t slots[kRingSlots];
};

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_DEVICE_RING_H_
