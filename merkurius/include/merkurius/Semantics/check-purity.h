//===-- include/merkurius/Semantics/check-purity.h --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_CHECK_PURITY_H_
#define MERKURIUS_SEMANTICS_CHECK_PURITY_H_

// Purity checking (design document, "Semantic analysis", pass 3; reference
// manual, "Impurity declarations"): each call of an impure or semipure
// predicate, and each application of an impure or semipure function, is
// annotated as such; and each predicate's or function's purity, the least
// of its goals', with promises, is that of its declaration.

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/check-types.h"
#include "merkurius/Semantics/resolve-names.h"
#include "merkurius/Semantics/semantic-form.h"

namespace merkurius::semantics {

// Reports the module's purity errors, by the callees that type checking
// chose; a goal with a type error is not checked.
void CheckPurity(const parser::ParseTree &, const parser::TokenSequence &,
    const SemanticForm &, const NameResolution &, const TypeChecking &,
    parser::Messages &);

} // namespace merkurius::semantics

#endif // MERKURIUS_SEMANTICS_CHECK_PURITY_H_
