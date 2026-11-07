//===-- runtime/sha256.h - SHA-256 ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SHA-256 (FIPS 180-4), for the fingerprints of message types (encode.cpp)
// and, later, the cookie's HMAC (docs/INPROGRESS.md, Remote handles and
// TCP). The runtime links no library of cryptography.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_SHA256_H_
#define MERKURIUS_RUNTIME_SHA256_H_

#include <cstddef>
#include <cstdint>

namespace merkurius::runtime {

class Sha256 {
public:
  static constexpr std::size_t kDigestBytes{32};

  Sha256();
  void Update(const void *data, std::size_t bytes);
  // The digest of what was given; the object is not used after.
  void Finish(std::uint8_t digest[kDigestBytes]);

private:
  void Block(const std::uint8_t block[64]);

  std::uint32_t state_[8];
  std::uint8_t buffer_[64];
  std::size_t buffered_{0};
  std::uint64_t length_{0};
};

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_SHA256_H_
