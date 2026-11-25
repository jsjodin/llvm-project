//===-- include/merkurius/Semantics/check-pragmas.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_CHECK_PRAGMAS_H_
#define MERKURIUS_SEMANTICS_CHECK_PRAGMAS_H_

// The rules of the pragmas of the reference manual's "Pragmas" that need
// the module's procedures, as mmc reports them:
//
// - obsolete and obsolete_proc name a predicate or function of the module,
//   and obsolete_proc one of its modes ("Obsolescence"); a call of an
//   obsolete one, or a closure of it, is warned of, with the replacements
//   the pragma suggests, unless the caller is obsolete too;
// - type_spec names a predicate or function of the module, or one of its
//   modes, and its substitution gives each variable of the declaration
//   one type that does not mention the variables it replaces ("Syntax and
//   semantics of type specialization pragmas"). The specialization itself
//   is a hint, which merkc's polymorphism does without.
//
// The pragmas that name a predicate by name and arity only, such as inline,
// are checked with purity (check-purity.cpp).

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/check-modes.h"
#include "merkurius/Semantics/resolve-names.h"

namespace merkurius::semantics {

void CheckPragmas(const parser::ParseTree &, const parser::TokenSequence &,
    const NameResolution &, const ModeChecking &, parser::Messages &);

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_CHECK_PRAGMAS_H_
