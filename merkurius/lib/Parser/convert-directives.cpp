//===-- lib/Parser/convert-directives.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The converter's rules for the module directives initialise, finalise and
// mutable (reference manual, "Modules"), for promises, and for solver types
// (reference manual, "Solver types"). The checks are the ones mmc makes
// while reading these declarations.

#include "convert-rules.h"

namespace merkurius::parser {

namespace {

// Reports that a declaration has the wrong number of arguments.
void SayArgumentCount(const Term &t, llvm::StringRef name,
    llvm::StringRef expected, ConvertState &state) {
  state.messages.Say(GetSource(t, state.tokens),
      "a ':- %s' declaration must have %s"_err_en_US, name.str(),
      expected.str());
}

// The attributes of a mutable (reference manual, "Module-local mutable
// variables"), with the conflicts mmc reports.
bool ConvertMutableAttributes(const Term &term, MutableDecl &into,
    llvm::SmallVectorImpl<const MutableForeignName *> &foreignNames,
    ConvertState &state) {
  const Term &t{StripParentheses(term)};
  const auto *list{llvm::dyn_cast<ListTerm>(&t)};
  if (!list || list->tail()) {
    SayExpected(state, t, "a list of attributes");
    return false;
  }
  bool ok{true};
  // The attributes given, by name, for the conflict checks.
  llvm::SmallVector<llvm::StringRef, 4> given;
  for (const Term *element : list->elements()) {
    const Term &a{StripParentheses(*element)};
    llvm::StringRef name{
        llvm::isa<NameTerm>(a) || llvm::isa<CompoundTerm>(a) ? FunctorName(a)
                                                             : ""};
    if (const CompoundTerm * c{AsCompound(a, "foreign_name", 2)}) {
      auto lang{Apply(language, *c->arguments()[0], state)};
      auto foreign{Apply(expect("a foreign name", stringLiteral),
          *c->arguments()[1], state)};
      if (!lang || !foreign) {
        ok = false;
        continue;
      }
      for (const MutableForeignName *other : foreignNames) {
        if (other->language == *lang) {
          // mmc reports two foreign names for one language as a conflict.
          state.messages.Say(GetSource(a, state.tokens),
              "the mutable already has a foreign name for this "
              "language"_err_en_US);
          ok = false;
        }
      }
      foreignNames.push_back(
          state.arena.Make<MutableForeignName>(a.tokens(), *lang, *foreign));
      continue;
    }
    bool known{llvm::isa<NameTerm>(a) &&
        llvm::is_contained(
            {"trailed", "untrailed", "attach_to_io_state", "constant",
                "thread_local"},
            name)};
    if (!known) {
      SayExpected(state, a,
          "a mutable attribute: 'trailed', 'untrailed', "
          "'attach_to_io_state', 'constant', 'thread_local' or "
          "'foreign_name(Lang, Name)'");
      ok = false;
      continue;
    }
    if (llvm::is_contained(given, name)) {
      state.messages.Say(GetSource(a, state.tokens),
          "the attribute '%s' is given more than once"_err_en_US, name.str());
      ok = false;
      continue;
    }
    // The manual: constant cannot be given with attach_to_io_state or an
    // explicit trailed, and thread_local cannot be given with trailed or
    // constant.
    static constexpr std::pair<llvm::StringLiteral, llvm::StringLiteral>
        conflicts[]{{"trailed", "untrailed"}, {"constant", "attach_to_io_state"},
            {"constant", "trailed"}, {"thread_local", "trailed"},
            {"thread_local", "constant"}};
    for (const auto &[x, y] : conflicts) {
      if ((name == x && llvm::is_contained(given, y)) ||
          (name == y && llvm::is_contained(given, x))) {
        state.messages.Say(GetSource(a, state.tokens),
            "the attributes '%s' and '%s' conflict"_err_en_US,
            (name == x ? y : x).str(), name.str());
        ok = false;
      }
    }
    given.push_back(name);
    if (name == "trailed" || name == "untrailed") {
      into.trailed = name == "trailed";
    } else if (name == "attach_to_io_state") {
      into.attachToIoState = true;
    } else if (name == "constant") {
      into.constant = true;
    } else {
      into.threadLocal = true;
    }
  }
  return ok;
}

} // namespace

std::optional<const MutableDecl *> MutableRule::Convert(
    const Term &t, ConvertState &state) const {
  llvm::ArrayRef<const Term *> args{llvm::cast<CompoundTerm>(t).arguments()};
  if (args.size() != 4 && args.size() != 5) {
    SayArgumentCount(t, "mutable", "four or five arguments", state);
    return std::nullopt;
  }
  const Term &nameTerm{StripParentheses(*args[0])};
  bool ok{true};
  if (!llvm::isa<NameTerm>(nameTerm)) {
    SayExpected(state, nameTerm, "a mutable name");
    ok = false;
  }
  std::optional<Type> ty{Apply(type, *args[1], state)};
  std::optional<DataTerm> initial{Apply(dataTerm, *args[2], state)};
  std::optional<Inst> varInst{Apply(inst, *args[3], state)};
  MutableDecl decl{t.tokens(), {}, {}, {}, {}, std::nullopt, false, false,
      false, {}};
  llvm::SmallVector<const MutableForeignName *, 2> foreignNames;
  if (args.size() == 5) {
    ok &= ConvertMutableAttributes(*args[4], decl, foreignNames, state);
  }
  if (!ok || !ty || !initial || !varInst) {
    return std::nullopt;
  }
  decl.name = state.arena.Save(FunctorName(nameTerm));
  decl.type = *ty;
  decl.initial = *initial;
  decl.inst = *varInst;
  decl.foreignNames = state.arena.SaveArray(
      llvm::ArrayRef<const MutableForeignName *>{foreignNames});
  return state.arena.Make<MutableDecl>(decl);
}

std::optional<Item> InitialiseRule::Convert(
    const Term &t, ConvertState &state) const {
  const auto &c{llvm::cast<CompoundTerm>(t)};
  llvm::StringRef name{FunctorName(c)};
  InitialiseKind kind{name.starts_with("init") ? InitialiseKind::Initialise
                                               : InitialiseKind::Finalise};
  // mmc: "expected `predname/arity'", and it rejects pred(Name/Arity).
  if (auto pred{Apply(
          expect("'Name/Arity'", nameAndArity), *c.arguments()[0], state)}) {
    return state.arena.Make<InitialiseDecl>(t.tokens(), kind, *pred);
  }
  return std::nullopt;
}

std::optional<Item> PromiseRule::Convert(
    const Term &t, ConvertState &state) const {
  if (auto promised{Apply(goal, *llvm::cast<CompoundTerm>(t).arguments()[0],
          state)}) {
    return state.arena.Make<PromiseDecl>(t.tokens(), *promised);
  }
  return std::nullopt;
}

std::optional<PromiseDisjunctionKind> PromiseDisjunctionRule::Kind(
    const Term &t) {
  if (!llvm::isa<CompoundTerm>(t) || Arity(t) != 1) {
    return std::nullopt;
  }
  return llvm::StringSwitch<std::optional<PromiseDisjunctionKind>>(
      FunctorName(t))
      .Case("promise_exclusive", PromiseDisjunctionKind::Exclusive)
      .Case("promise_exhaustive", PromiseDisjunctionKind::Exhaustive)
      .Case("promise_exclusive_exhaustive",
          PromiseDisjunctionKind::ExclusiveExhaustive)
      .Default(std::nullopt);
}

bool PromiseDisjunctionRule::Matches(const Term &t) const {
  if (Kind(t)) {
    return true;
  }
  const auto *c{llvm::dyn_cast<CompoundTerm>(&t)};
  return c && Arity(*c) == 2 &&
      (FunctorName(*c) == "all" || FunctorName(*c) == "some") &&
      Kind(StripParentheses(*c->arguments()[1]));
}

std::optional<Item> PromiseDisjunctionRule::Convert(
    const Term &t, ConvertState &state) const {
  const CompoundTerm *all{AsCompound(t, "all", 2)};
  const Term &promise{
      all ? StripParentheses(*all->arguments()[1]) : t};
  const Term &body{
      AsCompound(t, "some", 2)
          ? StripParentheses(*llvm::cast<CompoundTerm>(t).arguments()[1])
          : promise};
  PromiseDisjunctionKind kind{*Kind(body)};
  llvm::StringRef name{FunctorName(body)};
  if (!all) {
    // mmc: "a `:- promise_exclusive' declaration must have the form
    // `:- all [<vars>] promise_exclusive ( <disjunction> )'".
    state.messages.Say(GetSource(t, state.tokens),
        "a ':- %s' declaration must have the form ':- all [Vars] %s "
        "Disjunction'"_err_en_US,
        name.str(), name.str());
    return std::nullopt;
  }
  auto vars{Apply(quantifiedVars, *all->arguments()[0], state)};
  auto promised{
      Apply(goal, *llvm::cast<CompoundTerm>(promise).arguments()[0], state)};
  if (!vars || !promised) {
    return std::nullopt;
  }
  const auto *const *disjunction{std::get_if<const Disjunction *>(&*promised)};
  if (!disjunction) {
    // mmc: "goal in declaration is not a disjunction".
    state.messages.Say(GetSource(promise, state.tokens),
        "the goal of a ':- %s' declaration must be a disjunction"_err_en_US,
        name.str());
    return std::nullopt;
  }
  return state.arena.Make<PromiseDisjunctionDecl>(
      t.tokens(), kind, *vars, *disjunction);
}

namespace {

// A constraint_store: one mutable, or a list of them.
std::optional<llvm::ArrayRef<const MutableDecl *>> ConvertConstraintStore(
    const Term &term, ConvertState &state) {
  const Term &t{StripParentheses(term)};
  constexpr auto oneMutable{expect("a mutable declaration", MutableRule{})};
  if (llvm::isa<ListTerm>(t)) {
    return Apply(expect("a list of mutable declarations", listOf(oneMutable)),
        t, state);
  }
  if (auto converted{Apply(oneMutable, t, state)}) {
    return state.arena.SaveArray(llvm::ArrayRef<const MutableDecl *>{*converted});
  }
  return std::nullopt;
}

// The operands of ',' terms, looking through parentheses.
void FlattenCommas(const Term &term, llvm::SmallVectorImpl<const Term *> &into) {
  const Term &t{StripParentheses(term)};
  if (const CompoundTerm * comma{AsCompound(t, ",", 2)}) {
    FlattenCommas(*comma->arguments()[0], into);
    FlattenCommas(*comma->arguments()[1], into);
  } else {
    into.push_back(&t);
  }
}

} // namespace

std::optional<Item> SolverTypeRule::Convert(
    const Term &t, ConvertState &state) const {
  const Term &inner{StripParentheses(*llvm::cast<CompoundTerm>(t).arguments()[0])};
  const CompoundTerm *typeDecl{AsCompound(inner, "type", 1)};
  if (!typeDecl) {
    // mmc: "the `:- solver' keyword should be followed by a type
    // definition".
    state.messages.Say(GetSource(t, state.tokens),
        "':- solver' must be followed by a type declaration"_err_en_US);
    return std::nullopt;
  }
  const Term &body{StripParentheses(*typeDecl->arguments()[0])};
  if (AsCompound(body, "--->", 2)) {
    state.messages.Say(GetSource(t, state.tokens),
        "a solver type cannot have data constructors"_err_en_US);
    return std::nullopt;
  }
  if (AsCompound(body, "==", 2)) {
    state.messages.Say(GetSource(t, state.tokens),
        "a solver type cannot be an equivalence type"_err_en_US);
    return std::nullopt;
  }
  const CompoundTerm *where{AsCompound(body, "where", 2)};
  auto head{Apply(expect("a type", typeHead),
      where ? *where->arguments()[0] : body, state)};
  if (!where) {
    // :- solver type t(T)., an abstract solver type.
    if (!head) {
      return std::nullopt;
    }
    auto [name, parameters]{*head};
    return state.arena.Make<AbstractTypeDecl>(
        t.tokens(), name, parameters, nullptr, true);
  }
  // The attributes, which must come in this order (mmc: "attributes are
  // either badly ordered or contain an unrecognised attribute").
  static constexpr llvm::StringLiteral order[]{"representation", "ground",
      "any", "constraint_store", "equality", "comparison"};
  llvm::SmallVector<const Term *, 6> attributes;
  FlattenCommas(*where->arguments()[1], attributes);
  const Term *values[std::size(order)]{};
  std::size_t next{0};
  bool ok{head.has_value()};
  for (const Term *attribute : attributes) {
    const CompoundTerm *is{AsCompound(*attribute, "is", 2)};
    const Term *key{is ? &StripParentheses(*is->arguments()[0]) : nullptr};
    std::size_t j{next};
    while (key && llvm::isa<NameTerm>(*key) && j < std::size(order) &&
        FunctorName(*key) != order[j]) {
      ++j;
    }
    if (!key || !llvm::isa<NameTerm>(*key) || j == std::size(order)) {
      SayExpected(state, *attribute,
          "a solver type attribute, in the order 'representation is', "
          "'ground is', 'any is', 'constraint_store is', 'equality is', "
          "'comparison is'");
      ok = false;
      break;
    }
    values[j] = is->arguments()[1];
    next = j + 1;
  }
  if (ok && !values[0]) {
    state.messages.Say(GetSource(t, state.tokens),
        "a solver type definition must have a 'representation' "
        "attribute"_err_en_US);
    ok = false;
  }
  if (!ok) {
    return std::nullopt;
  }
  std::optional<Type> representation{Apply(type, *values[0], state)};
  ok = representation.has_value();
  std::optional<Inst> ground, any;
  ok &= ConvertOptional(inst, values[1], ground, state);
  ok &= ConvertOptional(inst, values[2], any, state);
  llvm::ArrayRef<const MutableDecl *> store;
  if (values[3]) {
    auto converted{ConvertConstraintStore(*values[3], state)};
    ok &= converted.has_value();
    store = converted.value_or(store);
  }
  const SymName *equality{nullptr}, *comparison{nullptr};
  constexpr auto predName{expect("a predicate name", symName)};
  ok &= ConvertOptional(predName, values[4], equality, state);
  ok &= ConvertOptional(predName, values[5], comparison, state);
  if (!ok) {
    return std::nullopt;
  }
  auto [name, parameters]{*head};
  return state.arena.Make<SolverTypeDefn>(t.tokens(), name, parameters,
      *representation, ground, any, store, equality, comparison);
}

} // namespace merkurius::parser
