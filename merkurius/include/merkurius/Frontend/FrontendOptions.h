//===-- include/merkurius/Frontend/FrontendOptions.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_FRONTEND_FRONTENDOPTIONS_H_
#define MERKURIUS_FRONTEND_FRONTENDOPTIONS_H_

#include <string>
#include <vector>

namespace merkurius::frontend {

// The action -fc1 runs on each input.
enum class ActionKind {
  // -E: print the prescanner's output. Until the prescanner exists, this is
  // the input unchanged.
  PrintPreprocessedInput,
};

struct FrontendOptions {
  ActionKind programAction{ActionKind::PrintPreprocessedInput};
  std::vector<std::string> inputs;
  // Empty or "-" means standard output.
  std::string outputFile;
  bool showHelp{false};
  bool showVersion{false};
};

struct LangOptions {
  // -fopenmp: read %$omp directive lines instead of skipping them as
  // comments.
  bool openMP{false};
  // -fopenmp-version=, as in Clang and Flang: 52 for OpenMP 5.2.
  unsigned openMPVersion{52};
};

} // namespace merkurius::frontend

#endif // MERKURIUS_FRONTEND_FRONTENDOPTIONS_H_
