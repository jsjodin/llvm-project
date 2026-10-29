//===-- include/merkurius/Driver/Options.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_DRIVER_OPTIONS_H_
#define MERKURIUS_DRIVER_OPTIONS_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include <optional>

namespace merkurius::options {

// Which layer accepts an option. DriverOption is LLVM's DefaultVis.
enum MerkuriusVisibility {
  DriverOption = llvm::opt::DefaultVis,
  FC1Option = (1 << 1),
};

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "merkurius/Driver/Options.inc"
  LastOption
#undef OPTION
};

const llvm::opt::OptTable &getDriverOptTable();

// Parses `args` with the options that `visibility` accepts. Missing values
// and unknown arguments are reported to standard error, with a suggestion
// when one is close; the result is then std::nullopt.
std::optional<llvm::opt::InputArgList> parseArgs(
    llvm::ArrayRef<const char *> args, MerkuriusVisibility visibility);

// Reports an error to standard error, prefixed with "merkc: error: ".
void reportError(const llvm::Twine &message);

} // namespace merkurius::options

#endif // MERKURIUS_DRIVER_OPTIONS_H_
