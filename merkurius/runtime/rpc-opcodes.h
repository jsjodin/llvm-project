//===-- runtime/rpc-opcodes.h - The device's RPC opcodes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The opcodes that the device's runtime (device/device.cpp) sends over the
// offload plugin's RPC server, LLVM libc's protocol (libc/shared/rpc.h), and
// that the host's callback (offload.cpp) handles. libc's are 'c' << 24 and
// up, the offload runtime's 'o' << 24 and up; ours are 'm' << 24 and up
// (docs/INPROGRESS.md, The device's side of the first slice, Decision 1).
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_RPC_OPCODES_H_
#define MERKURIUS_RUNTIME_RPC_OPCODES_H_

#include <stdint.h>

namespace merkurius::runtime {

constexpr uint32_t RpcOpcode(uint32_t n) { return ('m' << 24) | n; }

// A send of a one-word message: the buffer's words are the handle's five,
// the message type's key, under which the host registered its type_info,
// and the message. The host
// answers once it has queued it.
constexpr uint32_t rpcSend{RpcOpcode(1)};

// The ring of a mailbox the device receives from (device-ring.h): the
// buffer's words are the handle's five, the device's number and what the
// messages are (kRingOfWords, kRingOfArrays or kRingOfTerms). The host answers
// with the ring's address and its head, or 0 if the mailbox is closed or
// another node's.
constexpr uint32_t rpcAttach{RpcOpcode(2)};

// A chunk for the device's arena: the buffer's words are the device's
// number, the address of its arena, and the bytes the chunk must hold at
// least. The host answers with the chunk's address and its size in bytes.
// It frees the chunks, and clears the arena, when no region is running.
constexpr uint32_t rpcChunk{RpcOpcode(3)};

// An array the device made, which a region gives, copied by the device into
// chunks of host memory (rpcPinnedChunk): the buffer's words are the copy,
// its header, which has its size and its elements' width (array.h), and the
// copy's first chunk. The host copies it again
// and answers with that copy's address, and frees the chunks.
constexpr uint32_t rpcGiveArray{RpcOpcode(4)};

// 8 and 9 were output, which is libc's stdio now (design document, "Output
// from a region").

// 5 was a send of an array that the host copied from the device's memory,
// which the device's caches may not have written back while it runs: the
// device copies the array into host memory (rpcSendTerm).

// The messages of a mailbox that did not fit in its ring: the buffer's
// words are the handle's five. The host moves what fits into the ring, and
// then answers.
constexpr uint32_t rpcDrain{RpcOpcode(6)};

// The oldest message of a mailbox of arrays, whose header its ring gave, as
// rpcTakeTerm takes a term: the host copies it into host memory that the
// device reads, rather than into the device's memory, which the device's
// caches may hold stale while it runs. The device copies it into its arena
// and then moves the ring's head, after which the host frees the copy.
constexpr uint32_t rpcTakeArray{RpcOpcode(7)};

// An error on the device that ends the program: the buffer's first word is
// what the text is, 0 for an uncaught exception's, 1 for a feature not yet
// implemented and 2 for a limit of the runtime, and then the text, by
// send_n. The host reports it as it
// reports the same on the host, and ends the program, so it does not answer.
// Host memory that the device writes and the host reads, for a copy of a
// message (copy.h): the buffer's words are the device's number and the
// bytes the chunk must hold at least. The host answers with the chunk's
// address. Its first word is for the next chunk of the copy, and its second
// is the device's number; its size in bytes is its third.
constexpr uint32_t rpcPinnedChunk{RpcOpcode(8)};

// A send of a message that the device copied into such chunks: the buffer's
// words are the handle's five, the message type's key, the copy, and the
// copy's first chunk. The host answers once it has queued the message, and
// frees the chunks.
constexpr uint32_t rpcSendTerm{RpcOpcode(9)};

// The oldest message that the device has not taken from a ring of terms,
// copied by the host into host memory that the device reads: the buffer's
// words are the handle's five. The host answers with whether it found the
// message, which it does not if the mailbox is closed, and the copy. The device
// copies it, by the message type's type_info, into its arena, and then moves
// the ring's head, after which the host frees the copy.
constexpr uint32_t rpcTakeTerm{RpcOpcode(11)};

constexpr uint32_t rpcFail{RpcOpcode(10)};

// How long the teams of a region with team mailboxes wait for another of
// them to start before the region is reported, which a team asks once it
// has waited a millisecond: the host answers with the milliseconds that
// MERK_TEAMS_STALL_MILLISECONDS gives, or 10000.
constexpr uint32_t rpcTeamsStall{RpcOpcode(12)};

// A GPU process has ended, in the kernel of its module's processes
// (place.h): the buffer's words are the device's number, the kernel's
// table and the process's slot, which the device has set idle. The host
// answers once it has taken the process off the slot, which it may then
// give another.
constexpr uint32_t rpcProcessEnded{RpcOpcode(13)};

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_RPC_OPCODES_H_
