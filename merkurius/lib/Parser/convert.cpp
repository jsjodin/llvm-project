//===-- lib/Parser/convert.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/convert.h"
#include "merkurius/Parser/message.h"
#include "merkurius/Parser/term-combinators.h"
#include "merkurius/Parser/term-reader.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include <vector>

namespace merkurius::parser {

//===----------------------------------------------------------------------===//
// Helpers for the combinators
//===----------------------------------------------------------------------===//

const Term &StripParentheses(const Term &term) {
  const Term *t{&term};
  while (const auto *parenthesized{llvm::dyn_cast<ParenthesizedTerm>(t)}) {
    t = parenthesized->inner();
  }
  return *t;
}

llvm::StringRef FunctorName(const Term &t) {
  if (const auto *name{llvm::dyn_cast<NameTerm>(&t)}) {
    return name->value();
  } else if (const auto *compound{llvm::dyn_cast<CompoundTerm>(&t)}) {
    return compound->functor()->value();
  }
  return {};
}

std::size_t Arity(const Term &t) {
  if (const auto *compound{llvm::dyn_cast<CompoundTerm>(&t)}) {
    return compound->arguments().size();
  }
  return 0;
}

// The first line of a term's source, shortened to fit in a message.
static std::string Excerpt(const Term &t, const TokenSequence &tokens) {
  constexpr std::size_t maxLength{40};
  llvm::StringRef source{GetSource(t, tokens)};
  llvm::StringRef line{source.take_until([](char c) { return c == '\n'; })};
  if (line.size() < source.size() || line.size() > maxLength) {
    return line.take_front(maxLength).rtrim().str() + "...";
  }
  return line.str();
}

void SayExpected(ConvertState &state, const Term &t, llvm::StringRef expected) {
  state.messages.Say(GetSource(t, state.tokens),
      "expected %s, but found '%s'"_err_en_US, expected.str(),
      Excerpt(t, state.tokens));
}

// A qualified name is the term '.'(Qualifier, Name), with the qualifiers
// nested on the left, or a name with '__' in it.
static bool IsQualified(const Term &t) {
  return llvm::isa<CompoundTerm>(t) && FunctorName(t) == "." && Arity(t) == 2;
}

bool SymNameRule::Matches(const Term &t) const {
  return llvm::isa<NameTerm>(t) || IsQualified(t);
}

// Splits a__b__c into a, b and c, but leaves a name that begins or ends
// with '__', or has '__' in it with nothing between, as it is.
static void SplitUnderscores(
    llvm::StringRef name, llvm::SmallVectorImpl<llvm::StringRef> &parts) {
  llvm::SmallVector<llvm::StringRef, 4> split;
  name.split(split, "__");
  if (llvm::any_of(split, [](llvm::StringRef s) { return s.empty(); })) {
    parts.push_back(name);
  } else {
    parts.append(split.begin(), split.end());
  }
}

std::optional<const SymName *> SymNameRule::Convert(
    const Term &t, ConvertState &state) const {
  // The parts, outermost first: '.'('.'(a, b), c) is a, b, c.
  llvm::SmallVector<llvm::StringRef, 4> parts;
  bool ok{true};
  auto collect{[&](const Term &part, auto &self) -> void {
    const Term &p{StripParentheses(part)};
    if (IsQualified(p)) {
      const auto &qualified{llvm::cast<CompoundTerm>(p)};
      self(*qualified.arguments()[0], self);
      self(*qualified.arguments()[1], self);
    } else if (const auto *name{llvm::dyn_cast<NameTerm>(&p)}) {
      SplitUnderscores(name->value(), parts);
    } else {
      SayExpected(state, p, "a name");
      ok = false;
    }
  }};
  collect(t, collect);
  if (!ok) {
    return std::nullopt;
  }
  for (llvm::StringRef &part : parts) {
    part = state.arena.Save(part);
  }
  llvm::ArrayRef<llvm::StringRef> all{parts};
  return state.arena.Make<SymName>(t.tokens(),
      state.arena.SaveArray(all.drop_back()), all.back());
}

//===----------------------------------------------------------------------===//
// The module system (reference manual, "The module system")
//===----------------------------------------------------------------------===//

static constexpr auto moduleName{expect("a module name", symName)};

static constexpr auto moduleDecl{
    construct<ModuleDecl>(functor("module", moduleName))};

static constexpr auto endModuleDecl{
    construct<EndModuleDecl>(functor("end_module", moduleName))};

static constexpr auto sectionDecl{construct<SectionDecl>(
    anyOf(withValue(Section::Interface, functor("interface")),
        withValue(Section::Implementation, functor("implementation"))))};

static constexpr auto importDecl{construct<ImportDecl>(anyOf(
    withValue(ImportKind::ImportModule,
        functor("import_module", commaListOf(moduleName))),
    withValue(
        ImportKind::UseModule, functor("use_module", commaListOf(moduleName))),
    withValue(ImportKind::IncludeModule,
        functor("include_module", commaListOf(moduleName)))))};

//===----------------------------------------------------------------------===//
// Items (reference manual, "Items")
//===----------------------------------------------------------------------===//

static constexpr auto declaration{
    anyOf<Item>(moduleDecl, endModuleDecl, sectionDecl, importDecl)};

// Whether `name` is a declaration of the reference manual's list
// ("Declarations") that is not converted yet.
static bool IsDeclarationToDo(llvm::StringRef name) {
  return llvm::StringSwitch<bool>(name)
      .Cases({"type", "solver", "pred", "func", "inst", "mode"}, true)
      .Cases({"typeclass", "instance", "pragma", "promise"}, true)
      .Cases({"initialise", "initialize", "finalise", "finalize"}, true)
      .Case("mutable", true)
      .Default(false);
}

std::optional<Item> ConvertItem(const Term &term, ConvertState &state) {
  const Term &t{StripParentheses(term)};
  if (!(llvm::isa<CompoundTerm>(t) && FunctorName(t) == ":-" &&
          Arity(t) == 1)) {
    // A rule, a DCG rule or a fact.
    state.messages.Say(GetSource(t, state.tokens), "clauses"_todo_en_US);
    return std::nullopt;
  }
  const Term &decl{
      StripParentheses(*llvm::cast<CompoundTerm>(t).arguments()[0])};
  if (declaration.Matches(decl)) {
    return declaration.Convert(decl, state);
  }
  if (IsDeclarationToDo(FunctorName(decl))) {
    state.messages.Say(GetSource(decl, state.tokens),
        "the ':- %s' declaration"_todo_en_US, FunctorName(decl).str());
    return std::nullopt;
  }
  SayExpected(state, decl, "a declaration");
  return std::nullopt;
}

const ParseTree &ConvertFile(
    const TokenSequence &tokens, ParseTreeArena &arena, Messages &messages) {
  // An item with a lexical error is not converted: the reader may have
  // read a term from it, but not the one that was meant.
  std::vector<const char *> lexerErrors;
  for (const Message &message : messages.messages()) {
    if (const CharBlock * at{message.GetCharBlock()}) {
      lexerErrors.push_back(at->begin());
    }
  }
  TermArena terms;
  TermReader reader{tokens, terms, messages};
  ConvertState state{tokens, arena, messages};
  std::vector<Item> items;
  while (std::optional<ItemTerm> item{reader.ReadItem()}) {
    const Token &first{tokens.tokens[item->tokens.first]};
    const Token &last{tokens.tokens[item->tokens.last]};
    bool lexerError{llvm::any_of(lexerErrors, [&](const char *at) {
      return at >= first.spelling.begin() && at <= last.spelling.end();
    })};
    if (item->term && !lexerError) {
      if (std::optional<Item> converted{ConvertItem(*item->term, state)}) {
        items.push_back(*converted);
      }
    }
    terms.Reset();
  }
  auto lastToken{static_cast<std::uint32_t>(tokens.tokens.size() - 1)};
  return *arena.Make<ParseTree>(TokenRange{0, lastToken},
      arena.SaveArray(llvm::ArrayRef<Item>{items}));
}

} // namespace merkurius::parser
