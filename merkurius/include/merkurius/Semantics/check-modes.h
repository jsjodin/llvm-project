//===-- include/merkurius/Semantics/check-modes.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_CHECK_MODES_H_
#define MERKURIUS_SEMANTICS_CHECK_MODES_H_

// Mode analysis (design document, "Semantic analysis", pass 4). So far it
// reads the module's inst and mode definitions and its predicates' and
// functions' mode declarations into procedures (reference manual,
// "Predicate and function mode declarations"), and reports the errors in
// them that mmc reports.

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/check-types.h"
#include "merkurius/Semantics/insts.h"
#include "merkurius/Semantics/resolve-names.h"
#include "merkurius/Semantics/semantic-form.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>
#include <optional>
#include <vector>

namespace merkurius::semantics {

// One mode of a predicate or function: from a mode declaration, a
// predicate or function declaration that combines types and modes, or,
// for a function without mode declarations, the default mode, with every
// argument in and the result out (reference manual, "Predicate and
// function mode declarations").
struct Procedure {
  const Symbol *symbol;
  unsigned number; // in the order of the declarations
  llvm::ArrayRef<ArgMode> modes; // a function's result last
  std::optional<parser::Determinism> determinism;
  // The name in the declaration; empty for a default mode.
  parser::CharBlock source;
  // The inst parameters of the declaration, by number (reference manual,
  // "Constrained polymorphic modes").
  llvm::ArrayRef<const Inst *> parameters{};
};

class ModeChecking {
public:
  ModeChecking();
  ~ModeChecking();
  InstContext &insts() { return insts_; }
  // The procedures of the module's predicates and functions, in the order
  // of their symbols.
  llvm::ArrayRef<Procedure> procedures() const { return procedures_; }
  llvm::ArrayRef<Procedure> ProceduresOf(const Symbol &) const;
  // The module's own inst definitions, with their bodies.
  llvm::ArrayRef<const Symbol *> instDefinitions() const { return instDefinitions_; }

private:
  friend class ModeChecker;
  common::Arena arena_;
  InstContext insts_;
  std::vector<Procedure> procedures_;
  llvm::DenseMap<const Symbol *, std::pair<std::size_t, std::size_t>>
      bySymbol_;
  std::vector<const Symbol *> instDefinitions_;
};

struct ModeCheckOptions {
  // Infer the modes of predicates without mode declarations, rather than
  // report each (mmc's --infer-modes).
  bool inferModes{false};
  // At most this many passes of mode inference (mmc's
  // --mode-inference-iteration-limit).
  unsigned inferenceIterationLimit{30};
};

std::unique_ptr<ModeChecking> CheckModes(const parser::ParseTree &,
    const parser::TokenSequence &, const SemanticForm &,
    const NameResolution &, TypeChecking &, parser::Messages &,
    const ModeCheckOptions & = {});

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_CHECK_MODES_H_
