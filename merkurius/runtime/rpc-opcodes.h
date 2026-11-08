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

// A poll of a mailbox: the buffer's words are the handle's five. The host
// answers with 1 and the oldest message, which it removes, or 0 if the
// mailbox has none, or is closed or another node's.
constexpr uint32_t rpcPoll{RpcOpcode(2)};

// A chunk for the device's arena: the buffer's words are the device's
// number, the address of its arena, and the bytes the chunk must hold at
// least. The host answers with the chunk's address and its size in bytes.
// It frees the chunks, and clears the arena, when no region is running.
constexpr uint32_t rpcChunk{RpcOpcode(3)};

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_RPC_OPCODES_H_
