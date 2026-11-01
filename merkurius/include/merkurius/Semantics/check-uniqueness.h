//===-- include/merkurius/Semantics/check-uniqueness.h ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_CHECK_UNIQUENESS_H_
#define MERKURIUS_SEMANTICS_CHECK_UNIQUENESS_H_

// Uniqueness checking (design document, "Semantic analysis", pass 6): the
// second phase of unique mode analysis, which needs to know where
// backtracking can happen, and so runs after determinism analysis, on the
// bodies it built (Overton 2003, section 3.2). "A value is not
// considered `unique' if it might be needed on backtracking" (reference
// manual, "Unique modes"): a unique value that a goal may need when it is
// run again on backtracking is mostly_unique there, so that a call that
// would update it destructively, or a head variable that must end unique,
// is a mode error, as in mmc. A value is needed so
//
// - after a goal that can succeed more than once, if it was bound before
//   the goal and is used after it, since backtracking into the goal runs
//   what follows again, or if a call so made it, since the callee may need
//   it to find its next solution; where only the first solution is
//   wanted, only if a goal after it there can fail;
// - in a disjunction that can succeed more than once, if it was bound
//   before the disjunction and is used in it, as mmc makes it, since a
//   later branch may run after one has used it; where only its first
//   solution is wanted, or it has at most one, in a branch that can fail
//   and uses it;
// - in the condition of an if-then-else, if the else part uses it too;
// - in a negated goal, if it is used after the negation.
//
// A value made from one that is mostly_unique, by an assignment or a
// construction, or taken from one, by a deconstruction, is mostly_unique
// too, but for one the condition of an if-then-else made so, as in mmc.
// Goals whose end mode analysis does not reach are left out. Only
// procedures whose modes and determinisms were checked are, and only the
// first error of each is reported, as in mmc's first phase.

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/check-determinism.h"
#include "merkurius/Semantics/check-modes.h"

namespace merkurius::semantics {

void CheckUniqueness(const parser::TokenSequence &, ModeChecking &,
    const DeterminismChecking &, parser::Messages &);

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_CHECK_UNIQUENESS_H_
