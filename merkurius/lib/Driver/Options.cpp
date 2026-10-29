//===-- lib/Driver/Options.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Driver/Options.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm::opt;

namespace merkurius::options {

#define OPTTABLE_CODE
#include "merkurius/Driver/Options.inc"
#undef OPTTABLE_CODE

namespace {
class DriverOptTable : public OptTable {
public:
  DriverOptTable() : OptTable(optionTables()) {}
};
} // namespace

const OptTable &getDriverOptTable() {
  static DriverOptTable table;
  return table;
}

void reportError(const llvm::Twine &message) {
  llvm::WithColor::error(llvm::errs(), "merkc") << message << '\n';
}

std::optional<InputArgList> parseArgs(
    llvm::ArrayRef<const char *> args, MerkuriusVisibility visibility) {
  const OptTable &table{getDriverOptTable()};
  Visibility mask{static_cast<unsigned>(visibility)};
  unsigned missingIndex{0}, missingCount{0};
  InputArgList parsed{table.ParseArgs(args, missingIndex, missingCount, mask)};

  bool ok{true};
  if (missingCount) {
    reportError("argument to '" +
        llvm::Twine(parsed.getArgString(missingIndex)) +
        "' is missing (expected " + llvm::Twine(missingCount) + " value" +
        (missingCount == 1 ? "" : "s") + ")");
    ok = false;
  }
  for (const Arg *arg : parsed.filtered(OPT_UNKNOWN)) {
    std::string spelling{arg->getAsString(parsed)};
    std::string nearest;
    if (table.findNearest(spelling, nearest, mask) > 1) {
      reportError("unknown argument: '" + spelling + "'");
    } else {
      reportError("unknown argument '" + spelling + "'; did you mean '" +
          nearest + "'?");
    }
    ok = false;
  }
  if (!ok) {
    return std::nullopt;
  }
  return parsed;
}

} // namespace merkurius::options
