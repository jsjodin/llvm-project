//===-- include/merkurius/CodeGen/TargetOptions.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_CODEGEN_TARGETOPTIONS_H_
#define MERKURIUS_CODEGEN_TARGETOPTIONS_H_

// The options that choose the target (TargetInfo) and the optimization
// level.

#include <string>

namespace merkurius::codegen {

struct TargetOptions {
  std::string triple; // -triple; empty for the host's
  std::string cpu; // -target-cpu
  std::string features; // -target-feature, joined with commas
  unsigned optimizationLevel{0}; // -O
};

} // namespace merkurius::codegen

#endif // MERKURIUS_CODEGEN_TARGETOPTIONS_H_
