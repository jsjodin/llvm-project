//===-- lib/Semantics/expand-processes.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Process declarations' predicates as items (design document, "Processes
// and mailboxes"). The converter makes them (convert-processes.cpp); here
// they join the items, so that every later pass sees ordinary declarations
// and clauses. The declaration stays in the tree, after which they come.

#include "merkurius/Parser/message.h"
#include "merkurius/Semantics/desugar.h"
#include "llvm/ADT/STLExtras.h"
#include <vector>

namespace merkurius::semantics {

using namespace parser::literals;

namespace {

parser::CharBlock Source(
    const parser::TokenSequence &tokens, parser::TokenRange range) {
  return {tokens.tokens[range.first].spelling.begin(),
      tokens.tokens[range.last].spelling.end()};
}

// Reports what a declaration asks for that is not implemented yet.
void ReportUnimplemented(const parser::ProcessDecl &decl,
    const parser::TokenSequence &tokens, parser::Messages &messages) {
  parser::CharBlock at{Source(tokens, decl.tokens)};
  if (decl.receives.size() > 1) {
    messages.Say(at, "%s"_todo_en_US,
        "process kinds that receive several message types");
  }
}

} // namespace

const parser::ParseTree &ExpandProcesses(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, parser::ParseTreeArena &arena,
    parser::Messages &messages, bool clauses) {
  if (llvm::none_of(tree.items, [](const parser::Item &item) {
        return std::holds_alternative<const parser::ProcessDecl *>(item);
      })) {
    return tree;
  }
  // A clause made in the interface section waits for the implementation
  // section, or the end of the module.
  std::vector<parser::Item> items, waiting;
  bool inInterface{false}, named{false};
  for (const parser::Item &item : tree.items) {
    bool ends{std::holds_alternative<const parser::EndModuleDecl *>(item) ||
        (named && std::holds_alternative<const parser::ModuleDecl *>(item))};
    if (ends) {
      items.insert(items.end(), waiting.begin(), waiting.end());
      waiting.clear();
    }
    named |= std::holds_alternative<const parser::ModuleDecl *>(item);
    items.push_back(item);
    if (const auto *section{std::get_if<const parser::SectionDecl *>(&item)}) {
      inInterface = (*section)->section == parser::Section::Interface;
      if (!inInterface) {
        items.insert(items.end(), waiting.begin(), waiting.end());
        waiting.clear();
      }
    } else if (const auto *decl{
                   std::get_if<const parser::ProcessDecl *>(&item)}) {
      ReportUnimplemented(**decl, tokens, messages);
      items.push_back((*decl)->kindDecl);
      if ((*decl)->spawnDecl) {
        items.push_back((*decl)->spawnDecl);
      }
      if (clauses && (*decl)->spawnClause) {
        (inInterface ? waiting : items).push_back((*decl)->spawnClause);
      }
    }
  }
  items.insert(items.end(), waiting.begin(), waiting.end());
  return *arena.Make<parser::ParseTree>(
      tree.tokens, arena.SaveArray(llvm::ArrayRef<parser::Item>{items}));
}

} // namespace merkurius::semantics
