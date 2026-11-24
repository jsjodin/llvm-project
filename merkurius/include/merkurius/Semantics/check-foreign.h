//===-- include/merkurius/Semantics/check-foreign.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_CHECK_FOREIGN_H_
#define MERKURIUS_SEMANTICS_CHECK_FOREIGN_H_

// The rules of the foreign language interface's pragmas (reference manual,
// "Foreign language interface") that need the module's types, procedures
// and determinisms, and so run after determinism analysis, as mmc reports
// them:
//
// - a foreign_proc's procedure may not be multi or nondet ("pragma
//   foreign_proc"), and an erroneous one may not be will_not_throw_exception
//   ("Foreign code attributes");
// - a foreign_export's procedure may not be multi or nondet ("Calling
//   Merkurius from foreign code");
// - a foreign_type needs a Merkurius declaration or definition of its type,
//   abstract or a discriminated union, no wider visible than itself, and
//   neither a subtype nor the supertype of one ("Using foreign types from
//   Merkurius");
// - a foreign_enum is for an enumeration type the module defines, at most
//   one for each language, and maps each constructor to a value of its own
//   ("Using foreign enumerations in Merkurius code");
// - a foreign_export_enum is for an enumeration type, and its symbolic
//   names are distinct ("Using Merkurius enumerations in foreign code").
//
// The rules that need only the pragma, such as conflicting attributes, are
// the parser's; a pragma in the interface that may only be in the
// implementation is name resolution's.

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/check-determinism.h"
#include "merkurius/Semantics/check-modes.h"
#include "merkurius/Semantics/resolve-names.h"

namespace merkurius::semantics {

void CheckForeign(const parser::ParseTree &, const parser::TokenSequence &,
    const NameResolution &, const ModeChecking &, const DeterminismChecking &,
    parser::Messages &);

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_CHECK_FOREIGN_H_
