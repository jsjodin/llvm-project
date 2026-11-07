//===-- runtime/encode.h - The wire format ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A message's encoding for another node, and its type's fingerprint (design
// document, Wire format; docs/INPROGRESS.md, Remote handles and TCP).
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_ENCODE_H_
#define MERKURIUS_RUNTIME_ENCODE_H_

#include "descriptor.h"
#include "fragment.h"
#include <cstddef>
#include <cstdint>

namespace merkurius::runtime {

// Encoded bytes, from malloc, which the caller frees.
struct Encoded {
  std::uint8_t *bytes{nullptr};
  std::size_t size{0};
};

// A value of the type, encoded; null, or why it cannot be, with nothing to
// free.
const char *Encode(Word value, Word typeInfo, Encoded &encoded);

// A value of the type from its encoding, its cells made in the fragment;
// null, or why the bytes are not one. Nothing in the bytes can make it read
// outside them or allocate more than a small multiple of their size.
const char *Decode(const std::uint8_t *bytes, std::size_t size, Word typeInfo,
    Fragment &fragment, Word &value);

// The deepest a decoded value nests, but for the last argument of a
// constructor of the term's own type, as a list's tail, which is a loop.
constexpr int kMaxDecodeDepth{10000};

// A type's fingerprint: equal for two types exactly when their definitions,
// and those of every type they reach, are, to the chance of a collision of
// SHA-256 cut to 128 bits.
constexpr std::size_t kFingerprintBytes{16};
void Fingerprint(Word typeInfo, std::uint8_t fingerprint[kFingerprintBytes]);

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_ENCODE_H_
