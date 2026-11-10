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
// message a slot is the message, and for an array its size. Any other
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

// What a ring's mailbox holds: one-word messages, arrays of one-word
// elements, or other values, which are copied by their type_info (copy.h).
constexpr uint64_t kRingOfWords{0};
constexpr uint64_t kRingOfArrays{1};
constexpr uint64_t kRingOfTerms{2};

// The device's table of the rings it has, by the mailbox's slot, and its
// table of the target teams regions running on it, by the region's id
// (docs/INPROGRESS.md, Processes as teams, step 2), beside its arena's
// current chunk: the words that the host clears when no region is running.
constexpr uint64_t kRingCacheSlots{64};
constexpr uint64_t kTeamTableSlots{16};
constexpr uint64_t kArenaClearWords{1 + kRingCacheSlots + kTeamTableSlots};

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
