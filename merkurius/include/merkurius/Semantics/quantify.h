//===-- include/merkurius/Semantics/quantify.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_QUANTIFY_H_
#define MERKURIUS_SEMANTICS_QUANTIFY_H_

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/semantic-form.h"

namespace merkurius::semantics {

// Applies the reference manual's rules of implicit quantification
// ("Implicit quantification") to a desugared clause: a variable local to
// each of two parallel goals becomes two variables, a variable that some
// quantifies becomes a new one where the same name is used outside it, and
// every goal gets its nonlocal variables. Returns the quantified clause.
const ClauseForm &Quantify(const ClauseForm &, const parser::TokenSequence &,
    SemanticArena &, parser::Messages &);

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_QUANTIFY_H_
