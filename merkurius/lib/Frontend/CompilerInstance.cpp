//===-- lib/Frontend/CompilerInstance.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Frontend/CompilerInstance.h"
#include "merkurius/Driver/Options.h"

namespace merkurius::frontend {

llvm::raw_ostream *CompilerInstance::createDefaultOutputFile() {
  if (output_) {
    return output_.get();
  }
  llvm::StringRef path{getFrontendOpts().outputFile};
  if (path.empty()) {
    path = "-";
  }
  std::error_code ec;
  output_ = std::make_unique<llvm::raw_fd_ostream>(path, ec);
  if (ec) {
    options::reportError(
        "cannot open output file '" + path + "': " + ec.message());
    output_.reset();
    return nullptr;
  }
  return output_.get();
}

} // namespace merkurius::frontend
