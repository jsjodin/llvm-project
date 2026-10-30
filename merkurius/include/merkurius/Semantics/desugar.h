//===-- include/merkurius/Semantics/desugar.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_DESUGAR_H_
#define MERKURIUS_SEMANTICS_DESUGAR_H_

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/semantic-form.h"

namespace merkurius::semantics {

// Desugars the clauses of a parse tree into their semantic form. A clause
// with an error is reported and left out.
const SemanticForm &Desugar(const parser::ParseTree &,
    const parser::TokenSequence &, SemanticArena &, parser::Messages &);

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_DESUGAR_H_
