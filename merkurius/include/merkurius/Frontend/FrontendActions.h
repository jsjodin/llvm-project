//===-- include/merkurius/Frontend/FrontendActions.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Frontend actions, layered by how far they run, as Flang's are. Only the
// first exists so far; PrescanAction, PrescanAndParseAction and the rest are
// added with the stages they run.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_FRONTEND_FRONTENDACTIONS_H_
#define MERKURIUS_FRONTEND_FRONTENDACTIONS_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

namespace merkurius::frontend {

class CompilerInstance;

class FrontendAction {
public:
  virtual ~FrontendAction() = default;

  // Runs the action on one input file, or on standard input for "-".
  // Returns false after reporting an error.
  bool run(CompilerInstance &ci, llvm::StringRef input);

protected:
  virtual bool executeAction() = 0;

  CompilerInstance &getInstance() { return *instance_; }
  llvm::StringRef getInputName() const { return inputName_; }
  const llvm::MemoryBuffer &getInputBuffer() const { return *inputBuffer_; }

private:
  CompilerInstance *instance_{nullptr};
  llvm::StringRef inputName_;
  std::unique_ptr<llvm::MemoryBuffer> inputBuffer_;
};

// -E. Prints the input unchanged; it will print the prescanner's cooked
// stream once there is a prescanner.
class PrintPreprocessedInputAction : public FrontendAction {
  bool executeAction() override;
};

} // namespace merkurius::frontend

#endif // MERKURIUS_FRONTEND_FRONTENDACTIONS_H_
