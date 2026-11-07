//===-- runtime/node.h - Between mailboxes and connections ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the mailboxes (mailbox.cpp) and the connections to other nodes
// (node.cpp) give each other (docs/INPROGRESS.md, Remote handles and TCP).
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_NODE_H_
#define MERKURIUS_RUNTIME_NODE_H_

#include "descriptor.h"
#include "encode.h"
#include <cstddef>
#include <cstdint>

namespace merkurius::runtime {

// What a handle names: the node, the place, 0 for the host, in the top 16
// bits of the word whose other 48 are the slot, the generation and the
// secret.
struct Handle {
  Word nodeHigh;
  Word nodeLow;
  Word placeAndSlot;
  std::int64_t generation;
  Word secret;
};

constexpr unsigned kPlaceShift{48};
constexpr Word kSlotMask{(Word{1} << kPlaceShift) - 1};

// This node's id, chosen at random the first time it is asked for.
void ThisNode(Word &high, Word &low);

bool IsThisNode(Word high, Word low);

enum class Delivery {
  Delivered,
  // For a closed mailbox, a wrong secret, a place this node does not have,
  // or a node it is not connected to, as a local send to a closed mailbox
  // is dropped.
  Dropped,
  // Bytes that are not a value of the mailbox's type.
  Undecodable,
};

// mailbox.cpp: an encoded message for a mailbox of this node's host,
// decoded by the mailbox's own type and queued.
Delivery DeliverEncoded(
    const Handle &handle, const std::uint8_t *bytes, std::size_t size);

// mailbox.cpp: the mailbox registered under a name, for another node's
// lookup: 0 and its handle if its type's fingerprint is the one given, 1 if
// no mailbox has the name, 2 if its type is another.
std::int64_t LookupForPeer(const char *name,
    const std::uint8_t fingerprint[kFingerprintBytes], Handle &found);

// node.cpp: the one entry for an encoded message, which a remote send, a
// connection's reader and, later, a device's bridge call: a mailbox of this
// node's host gets it, one of another node goes to that node's connection,
// and one of a place this node does not have is dropped.
Delivery Route(
    const Handle &handle, const std::uint8_t *bytes, std::size_t size);

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_NODE_H_
