//===-- include/merkurius/Semantics/translate-loops.h -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_TRANSLATE_LOOPS_H_
#define MERKURIUS_SEMANTICS_TRANSLATE_LOOPS_H_

// The translation of loops (design document, "Loops", Meaning), after type
// and purity checking: each loop of a clause becomes a call of a new
// predicate, local to the module, its loop predicate, which calls another,
// its body predicate, in each iteration. Their types are the clause's,
// which type checking found, and their modes and determinism are inferred,
// as a predicate's without declarations are, so the later passes check
// the loop's sequential meaning as they check any predicate.

#include "merkurius/Parser/message.h"
#include "merkurius/Semantics/semantic-form.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

namespace merkurius::semantics {

class NameResolution;
struct Symbol;
class TypeChecking;

// The purity of each loop, its body's, as purity checking finds it.
using LoopPurities = llvm::DenseMap<const Loop *, parser::Purity>;

// The modules whose functions and predicates the translation calls,
// which the module need not import, as it need not import builtin: int,
// for the counters of for and count.
llvm::SmallVector<llvm::StringRef, 1> LoopModules(const SemanticForm &);

// A loop as translated, for code generation, which runs one with an
// OpenMP directive as an omp loop nest that calls its body predicate.
struct TranslatedLoop {
  const Loop *loop;
  const Symbol *loopPredicate;
  const Symbol *bodyPredicate;
  // For a loop with a for: its step, and the goal number in the clause of
  // the test that its trip count's computation starts with, Big < Small,
  // whose arguments are its bounds, Hi and Lo, or Lo and Hi for a negative
  // step. The for's counter is the loop predicate's first argument, and
  // its trip count its second.
  std::int64_t step{1};
  std::optional<unsigned> tripTest;
  // The values carried from one iteration to the next, those of each
  // fromto that does not end the loop and each threaded state variable:
  // the positions in the loop predicate's arguments of In's, or the
  // version in an iteration, and Last's, or the version after the loop.
  struct Threaded {
    unsigned in;
    unsigned after;
    const LoopIterator *fromto; // null for a state variable
  };
  std::vector<Threaded> threaded;
};

struct LoopTranslation {
  // The indices in types.clauses() of each translated clause, followed by
  // its loops' predicates' clauses, a loop's before those of the loops in
  // its body.
  std::vector<std::size_t> clauses;
  std::deque<TranslatedLoop> loops;
  // By loop predicate.
  llvm::DenseMap<const Symbol *, const TranslatedLoop *> byPredicate;
  const TranslatedLoop *Of(const Symbol &loopPredicate) const {
    return byPredicate.lookup(&loopPredicate);
  }
};

// Translates the loops of each clause that type checking checked without
// error, and whose loops purity checking checked: the clause's types then
// hold the translated clause, and the new predicates' clauses follow the
// others.
LoopTranslation TranslateLoops(NameResolution &, TypeChecking &,
    const LoopPurities &, const parser::TokenSequence &, SemanticArena &,
    parser::Messages &);

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_TRANSLATE_LOOPS_H_
