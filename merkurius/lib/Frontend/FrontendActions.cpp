//===-- lib/Frontend/FrontendActions.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Frontend/FrontendActions.h"
#include "merkurius/Driver/Options.h"
#include "merkurius/Frontend/CompilerInstance.h"

namespace merkurius::frontend {

bool FrontendAction::run(CompilerInstance &ci, llvm::StringRef input) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer{
      llvm::MemoryBuffer::getFileOrSTDIN(input, /*IsText=*/true)};
  if (!buffer) {
    options::reportError("cannot open input file '" + input +
        "': " + buffer.getError().message());
    return false;
  }
  instance_ = &ci;
  inputName_ = input;
  inputBuffer_ = std::move(*buffer);
  bool ok{executeAction()};
  inputBuffer_.reset();
  instance_ = nullptr;
  return ok;
}

bool PrintPreprocessedInputAction::executeAction() {
  llvm::raw_ostream *os{getInstance().createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  *os << getInputBuffer().getBuffer();
  os->flush();
  return true;
}

} // namespace merkurius::frontend
