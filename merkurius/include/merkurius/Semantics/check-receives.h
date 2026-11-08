//===-- include/merkurius/Semantics/check-receives.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_CHECK_RECEIVES_H_
#define MERKURIUS_SEMANTICS_CHECK_RECEIVES_H_

// The unselected-message warning (design document, "The receive goal"). A
// message that no arm of a receive goal selects stays queued, so messages
// can pile up. A predicate whose receive goals' arms for a mailbox, by the
// text of its term, leave out a constructor of the mailbox's message type
// is warned about once, at the first of those arms' mailbox. A constructor
// is selected by an arm whose pattern has it as its functor, whatever its
// arguments, and every message by an arm whose pattern is a variable of
// the arm's own. The values of a type without a known set of constructors,
// such as int, are selected only by the latter. It needs the mailboxes'
// types, so it runs after type checking, on the clauses without a type
// error; a predicate with one is left out.

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/check-types.h"
#include "merkurius/Semantics/insts.h"

namespace merkurius::semantics {

void CheckReceives(const parser::TokenSequence &, TypeChecking &,
    InstTypes &constructors, parser::Messages &);

} // namespace merkurius::semantics

#endif // MERKURIUS_SEMANTICS_CHECK_RECEIVES_H_
