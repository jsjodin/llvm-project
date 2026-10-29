//===-- lib/Parser/convert.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/convert.h"
#include "convert-rules.h"
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
std::string Excerpt(const Term &t, const TokenSequence &tokens) {
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

static bool IsQualified(const Term &t) {
  return llvm::isa<CompoundTerm>(t) && FunctorName(t) == "." && Arity(t) == 2;
}

void QualifiedParts(
    const Term &term, llvm::SmallVectorImpl<const Term *> &parts) {
  const Term &t{StripParentheses(term)};
  if (IsQualified(t)) {
    for (const Term *side : llvm::cast<CompoundTerm>(t).arguments()) {
      QualifiedParts(*side, parts);
    }
  } else {
    parts.push_back(&t);
  }
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

const SymName *MakeSymName(const Term &whole,
    llvm::ArrayRef<const Term *> qualifiers, llvm::StringRef name,
    ConvertState &state) {
  llvm::SmallVector<llvm::StringRef, 4> parts;
  bool ok{true};
  for (const Term *qualifier : qualifiers) {
    if (const auto *n{llvm::dyn_cast<NameTerm>(qualifier)}) {
      SplitUnderscores(n->value(), parts);
    } else {
      SayExpected(state, *qualifier, "a module name");
      ok = false;
    }
  }
  SplitUnderscores(name, parts);
  if (!ok) {
    return nullptr;
  }
  for (llvm::StringRef &part : parts) {
    part = state.arena.Save(part);
  }
  llvm::ArrayRef<llvm::StringRef> all{parts};
  return state.arena.Make<SymName>(StripParentheses(whole).tokens(),
      state.arena.SaveArray(all.drop_back()), all.back());
}

bool SymNameRule::Matches(const Term &t) const {
  return llvm::isa<NameTerm>(t) || IsQualified(t);
}

std::optional<const SymName *> SymNameRule::Convert(
    const Term &t, ConvertState &state) const {
  llvm::SmallVector<const Term *, 4> parts;
  QualifiedParts(t, parts);
  const Term &part{*parts.back()};
  if (!llvm::isa<NameTerm>(part)) {
    SayExpected(state, part, "a name");
    return std::nullopt;
  }
  if (const SymName *name{MakeSymName(
          t, llvm::ArrayRef(parts).drop_back(), FunctorName(part), state)}) {
    return name;
  }
  return std::nullopt;
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
// Types, insts and modes
//===----------------------------------------------------------------------===//

// The parts of a higher-order type or inst: pred(Args), (pred), func(Args)
// = Result or (func) = Result, with `prefix` "" or "any_".
struct HigherOrderShape {
  PredOrFunc kind;
  bool any;
  llvm::ArrayRef<const Term *> arguments;
  const Term *result; // for a function
};

static std::optional<HigherOrderShape> MatchHigherOrder(const Term &term) {
  const Term &t{StripParentheses(term)};
  const Term *head{&t};
  const Term *result{nullptr};
  if (const CompoundTerm * equals{AsCompound(t, "=", 2)}) {
    head = &StripParentheses(*equals->arguments()[0]);
    result = equals->arguments()[1];
  }
  llvm::StringRef name{FunctorName(*head)};
  bool any{name.consume_front("any_")};
  if (!(result ? name == "func" : name == "pred")) {
    return std::nullopt;
  }
  llvm::ArrayRef<const Term *> arguments;
  if (const auto *compound{llvm::dyn_cast<CompoundTerm>(head)}) {
    arguments = compound->arguments();
  }
  return HigherOrderShape{result ? PredOrFunc::Function : PredOrFunc::Predicate,
      any, arguments, result};
}

// pred(T1, ...), func(T1, ...) = T, and with modes and "is Det", the
// combined higher-order types; any of them semipure or impure.
class HigherOrderTypeRule {
public:
  using resultType = Type;
  bool Matches(const Term &t) const { return Split(t).has_value(); }
  std::optional<Type> Convert(const Term &t, ConvertState &state) const {
    Parts parts{*Split(t)};
    bool ok{true};
    std::optional<llvm::ArrayRef<const TypeAndMode *>> arguments{
        detail::ConvertElements(typeAndMode, parts.shape.arguments, state)};
    const TypeAndMode *result{nullptr};
    if (parts.shape.result) {
      auto converted{Apply(typeAndMode, *parts.shape.result, state)};
      ok = converted.has_value();
      result = converted.value_or(nullptr);
    }
    std::optional<Determinism> det;
    if (parts.determinism) {
      det = Apply(determinism, *parts.determinism, state);
      ok = ok && det.has_value();
    }
    if (!ok || !arguments) {
      return std::nullopt;
    }
    return state.arena.Make<HigherOrderType>(t.tokens(), parts.purity,
        parts.shape.kind, *arguments, result, det);
  }
  void Expected(llvm::raw_ostream &os) const { os << "a higher-order type"; }

private:
  struct Parts {
    Purity purity;
    HigherOrderShape shape;
    const Term *determinism;
  };
  static std::optional<Parts> Split(const Term &term) {
    const Term *t{&term};
    Purity purity{Purity::Pure};
    if (const CompoundTerm * p{AsCompound(*t, "impure", 1)}) {
      purity = Purity::Impure;
      t = p->arguments()[0];
    } else if (const CompoundTerm * p{AsCompound(*t, "semipure", 1)}) {
      purity = Purity::Semipure;
      t = p->arguments()[0];
    }
    const Term *det{nullptr};
    if (const CompoundTerm * is{AsCompound(*t, "is", 2)}) {
      t = is->arguments()[0];
      det = is->arguments()[1];
    }
    std::optional<HigherOrderShape> shape{MatchHigherOrder(*t)};
    if (!shape || shape->any) {
      return std::nullopt;
    }
    return Parts{purity, *shape, det};
  }
};

static constexpr auto typeAlternatives{expect("a type",
    anyOf<Type>(construct<TypeVar>(variable),
        construct<TupleType>(tupleOf(type)), HigherOrderTypeRule{},
        construct<NamedType>(appliedName(type))))};

std::optional<Type> TypeRule::Convert(
    const Term &t, ConvertState &state) const {
  return Apply(typeAlternatives, t, state);
}

std::optional<const TypeAndMode *> TypeAndModeRule::Convert(
    const Term &t, ConvertState &state) const {
  if (const CompoundTerm * colons{AsCompound(t, "::", 2)}) {
    std::optional<Type> ty{Apply(type, *colons->arguments()[0], state)};
    std::optional<Mode> m{Apply(mode, *colons->arguments()[1], state)};
    if (!ty || !m) {
      return std::nullopt;
    }
    return state.arena.Make<TypeAndMode>(t.tokens(), *ty, m);
  }
  if (std::optional<Type> ty{Apply(type, t, state)}) {
    return state.arena.Make<TypeAndMode>(t.tokens(), *ty);
  }
  return std::nullopt;
}

// The reserved inst names that take no arguments (reference manual,
// "Insts, modes, and mode definitions").
class BuiltinInstRule {
public:
  using resultType = const BuiltinInst *;
  bool Matches(const Term &t) const { return Lookup(t).has_value(); }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    return state.arena.Make<BuiltinInst>(t.tokens(), *Lookup(t));
  }
  void Expected(llvm::raw_ostream &os) const { os << "an inst"; }

private:
  static std::optional<BuiltinInstKind> Lookup(const Term &t) {
    if (!llvm::isa<NameTerm>(t)) {
      return std::nullopt;
    }
    using K = BuiltinInstKind;
    return llvm::StringSwitch<std::optional<K>>(FunctorName(t))
        .Case("free", K::Free)
        .Case("ground", K::Ground)
        .Case("unique", K::Unique)
        .Case("mostly_unique", K::MostlyUnique)
        .Case("clobbered", K::Clobbered)
        .Case("mostly_clobbered", K::MostlyClobbered)
        .Case("any", K::Any)
        .Case("unique_any", K::UniqueAny)
        .Case("mostly_unique_any", K::MostlyUniqueAny)
        .Case("clobbered_any", K::ClobberedAny)
        .Case("mostly_clobbered_any", K::MostlyClobberedAny)
        .Case("not_reached", K::NotReached)
        .Default(std::nullopt);
  }
};

std::optional<const Literal *> LiteralRule::Convert(
    const Term &t, ConvertState &state) const {
    LiteralKind kind{LiteralKind::String};
    if (const auto *integer{llvm::dyn_cast<IntegerTerm>(&t)}) {
      if (!integer->InRange()) {
        state.messages.Say(GetSource(t, state.tokens),
            "the integer '%s' is too large for its type"_err_en_US,
            Excerpt(t, state.tokens));
        return std::nullopt;
      }
      kind = LiteralKind::Integer;
    } else if (llvm::isa<FloatTerm>(t)) {
      kind = LiteralKind::Float;
    }
    std::string text;
    llvm::raw_string_ostream os{text};
    PrintCanonical(os, t);
    return state.arena.Make<Literal>(t.tokens(), kind, state.arena.Save(text));
}

// A function symbol of a bound inst: f(I1, I2), a, [] or [I | J], or a
// literal, which mmc accepts in an inst definition.
class BoundFunctorRule {
public:
  using resultType = const BoundFunctor *;
  bool Matches(const Term &t) const {
    if (const auto *list{llvm::dyn_cast<ListTerm>(&t)}) {
      return list->elements().empty() ||
          (list->elements().size() == 1 && list->tail());
    }
    return literal.Matches(t) || functor_.Matches(t);
  }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    if (const auto *list{llvm::dyn_cast<ListTerm>(&t)}) {
      llvm::StringRef name{list->elements().empty() ? "[]" : "[|]"};
      const SymName *symName{state.arena.Make<SymName>(
          t.tokens(), llvm::ArrayRef<llvm::StringRef>{}, name)};
      if (list->elements().empty()) {
        return state.arena.Make<BoundFunctor>(
            t.tokens(), symName, llvm::ArrayRef<Inst>{});
      }
      const Term *parts[]{list->elements()[0], list->tail()};
      if (auto arguments{detail::ConvertElements(
              inst, llvm::ArrayRef<const Term *>{parts}, state)}) {
        return state.arena.Make<BoundFunctor>(t.tokens(), symName, *arguments);
      }
      return std::nullopt;
    }
    if (literal.Matches(t)) {
      if (auto value{literal.Convert(t, state)}) {
        return state.arena.Make<BoundFunctor>(
            t.tokens(), *value, llvm::ArrayRef<Inst>{});
      }
      return std::nullopt;
    }
    return functor_.Convert(t, state);
  }
  void Expected(llvm::raw_ostream &os) const { os << "a function symbol"; }

private:
  static constexpr auto functor_{construct<BoundFunctor>(appliedName(inst))};
};

// bound(F1 ; F2 ; ...) and bound_unique(...).
class BoundInstRule {
public:
  using resultType = const BoundInst *;
  bool Matches(const Term &t) const {
    return AsCompound(t, "bound", 1) || AsCompound(t, "bound_unique", 1);
  }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    llvm::SmallVector<const BoundFunctor *, 8> functors;
    bool ok{true};
    auto collect{[&](const Term &term, auto &self) -> void {
      if (const CompoundTerm * semicolon{AsCompound(term, ";", 2)}) {
        for (const Term *side : semicolon->arguments()) {
          self(*side, self);
        }
      } else if (auto functor{Apply(BoundFunctorRule{}, term, state)}) {
        functors.push_back(*functor);
      } else {
        ok = false;
      }
    }};
    collect(*llvm::cast<CompoundTerm>(t).arguments()[0], collect);
    if (!ok) {
      return std::nullopt;
    }
    return state.arena.Make<BoundInst>(t.tokens(),
        FunctorName(t) == "bound_unique",
        state.arena.SaveArray(llvm::ArrayRef<const BoundFunctor *>{functors}));
  }
  void Expected(llvm::raw_ostream &os) const { os << "a bound inst"; }
};

// pred(Mode, ...) is Det and func(Mode, ...) = Mode is Det, and their
// any_pred and any_func forms (reference manual, "Builtin higher-order
// insts and modes").
class HigherOrderInstRule {
public:
  using resultType = const HigherOrderInst *;
  bool Matches(const Term &t) const {
    const CompoundTerm *is{AsCompound(t, "is", 2)};
    return is && MatchHigherOrder(*is->arguments()[0]).has_value();
  }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    const auto &is{llvm::cast<CompoundTerm>(t)};
    HigherOrderShape shape{*MatchHigherOrder(*is.arguments()[0])};
    auto arguments{detail::ConvertElements(mode, shape.arguments, state)};
    bool ok{arguments.has_value()};
    std::optional<Mode> result;
    if (shape.result) {
      result = Apply(mode, *shape.result, state);
      ok = ok && result.has_value();
    }
    std::optional<Determinism> det{
        Apply(determinism, *is.arguments()[1], state)};
    if (!ok || !det) {
      return std::nullopt;
    }
    return state.arena.Make<HigherOrderInst>(
        t.tokens(), shape.kind, shape.any, *arguments, result, *det);
  }
  void Expected(llvm::raw_ostream &os) const { os << "a higher-order inst"; }
};

static constexpr auto instAlternatives{expect("an inst",
    anyOf<Inst>(construct<InstVar>(variable),
        construct<InstVar>(functor("=<", variable, inst)), BuiltinInstRule{},
        BoundInstRule{}, HigherOrderInstRule{},
        construct<NamedInst>(appliedName(inst))))};

std::optional<Inst> InstRule::Convert(
    const Term &t, ConvertState &state) const {
  return Apply(instAlternatives, t, state);
}

// A variable is not a mode (mmc: "a variable such as `X' is not a valid
// mode"), so it is reported as one that is not expected.
static constexpr auto modeAlternatives{expect("a mode",
    anyOf<Mode>(construct<MappingMode>(functor(">>", inst, inst)),
        HigherOrderInstRule{}, construct<NamedMode>(appliedName(inst))))};

std::optional<Mode> ModeRule::Convert(
    const Term &t, ConvertState &state) const {
  return Apply(modeAlternatives, t, state);
}

//===----------------------------------------------------------------------===//
// Predicate and function declarations
//===----------------------------------------------------------------------===//

static constexpr auto classConstraint{
    expect("a type class constraint",
        construct<ClassConstraint>(appliedName(type)))};

static constexpr auto instConstraint{expect("an inst constraint such as 'I =< ground'",
    construct<InstConstraint>(functor("=<", variable, inst)))};

static constexpr auto universalConstraints{commaListOf(expect("a constraint",
    anyOf<Constraint>(instConstraint, classConstraint)))};

// The layers around pred(...), func(...) or mode(...), outermost first, as
// mmc accepts them: all [Vs], some [Vs], <= Constraints, => Constraints, and
// impure or semipure (reference manual, "Syntax for explicit type
// quantifiers", "Existential class constraints" and "Constrained
// polymorphic modes").
struct DeclarationLayers {
  const Term *universalVars{nullptr};
  const Term *existentialVars{nullptr};
  const Term *universalConstraints{nullptr};
  const Term *existentialConstraints{nullptr};
  Purity purity{Purity::Pure};
  PredOrFunc kind{PredOrFunc::Predicate};
  bool isMode{false};
  const Term *body{nullptr}; // the argument of pred, func or mode
};

static std::optional<DeclarationLayers> PeelDeclaration(const Term &term) {
  DeclarationLayers layers;
  const Term *t{&term};
  auto peel{[&](llvm::StringRef name, std::size_t arity, const Term *&into) {
    if (const CompoundTerm * c{AsCompound(*t, name, arity)}) {
      into = c->arguments()[0];
      t = c->arguments()[arity - 1];
    }
  }};
  peel("all", 2, layers.universalVars);
  peel("some", 2, layers.existentialVars);
  if (const CompoundTerm * c{AsCompound(*t, "<=", 2)}) {
    layers.universalConstraints = c->arguments()[1];
    t = c->arguments()[0];
  }
  if (const CompoundTerm * c{AsCompound(*t, "=>", 2)}) {
    layers.existentialConstraints = c->arguments()[1];
    t = c->arguments()[0];
  }
  if (const CompoundTerm * c{AsCompound(*t, "impure", 1)}) {
    layers.purity = Purity::Impure;
    t = c->arguments()[0];
  } else if (const CompoundTerm * c{AsCompound(*t, "semipure", 1)}) {
    layers.purity = Purity::Semipure;
    t = c->arguments()[0];
  }
  if (const CompoundTerm * c{AsCompound(*t, "pred", 1)}) {
    layers.body = c->arguments()[0];
  } else if (const CompoundTerm * c{AsCompound(*t, "func", 1)}) {
    layers.kind = PredOrFunc::Function;
    layers.body = c->arguments()[0];
  } else if (const CompoundTerm * c{AsCompound(*t, "mode", 1)}) {
    // Only inst constraints apply to a mode declaration, and "Name ==
    // Mode" defines a mode instead.
    if (layers.universalVars || layers.existentialVars ||
        layers.existentialConstraints || layers.purity != Purity::Pure ||
        AsCompound(*c->arguments()[0], "==", 2)) {
      return std::nullopt;
    }
    layers.isMode = true;
    layers.body = c->arguments()[0];
  } else {
    return std::nullopt;
  }
  return layers;
}

// The parts of a declaration's body: Head = Result `with_type` Type
// `with_inst` Inst is Det, where all but the head are optional.
struct DeclarationBody {
  const Term *head;
  const Term *result{nullptr};
  const Term *withType{nullptr};
  const Term *withInst{nullptr};
  const Term *determinism{nullptr};
};

static DeclarationBody SplitBody(const Term &body, bool isFunction) {
  DeclarationBody parts{&body};
  auto peel{[&](llvm::StringRef name, const Term *&into) {
    if (auto operands{BinaryOperands(*parts.head, name)}) {
      std::tie(parts.head, into) = *operands;
      return true;
    }
    return false;
  }};
  peel("is", parts.determinism);
  peel("with_inst", parts.withInst);
  // The : Type form is the older spelling of `with_type`.
  peel("with_type", parts.withType) || peel(":", parts.withType);
  if (isFunction) {
    peel("=", parts.result);
  }
  return parts;
}

static constexpr auto predicateHead{
    expect("a predicate or function name", appliedName(typeAndMode))};
static constexpr auto modeHead{
    expect("a predicate or function name", appliedName(mode))};

class PredDeclRule {
public:
  using resultType = const PredDecl *;
  bool Matches(const Term &t) const {
    std::optional<DeclarationLayers> layers{PeelDeclaration(t)};
    return layers && !layers->isMode;
  }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    DeclarationLayers layers{*PeelDeclaration(t)};
    bool isFunction{layers.kind == PredOrFunc::Function};
    DeclarationBody body{SplitBody(*layers.body, isFunction)};
    bool ok{true};
    llvm::ArrayRef<llvm::StringRef> universalVars, existentialVars;
    ok &= ConvertOptional(quantifiedVars, layers.universalVars,
        universalVars, state);
    ok &= ConvertOptional(quantifiedVars, layers.existentialVars,
        existentialVars, state);
    std::optional<std::tuple<const SymName *,
        llvm::ArrayRef<const TypeAndMode *>>>
        head{Apply(predicateHead, *body.head, state)};
    ok &= head.has_value();
    const TypeAndMode *result{nullptr};
    if (isFunction && !body.result) {
      SayExpected(state, StripParentheses(*body.head),
          "a function's arguments and result, as in 'f(int) = int'");
      ok = false;
    }
    ok &= ConvertOptional(typeAndMode, body.result, result, state);
    std::optional<Type> withType;
    std::optional<Inst> withInst;
    std::optional<Determinism> det;
    ok &= ConvertOptional(type, body.withType, withType, state);
    ok &= ConvertOptional(inst, body.withInst, withInst, state);
    ok &= ConvertOptional(determinism, body.determinism, det, state);
    llvm::ArrayRef<const ClassConstraint *> existentialConstraints;
    llvm::ArrayRef<Constraint> constraints;
    ok &= ConvertOptional(commaListOf(classConstraint),
        layers.existentialConstraints, existentialConstraints, state);
    ok &= ConvertOptional(universalConstraints, layers.universalConstraints,
        constraints, state);
    if (!ok) {
      return std::nullopt;
    }
    auto [name, arguments]{*head};
    // mmc: "some but not all arguments have modes".
    std::size_t withModes{0}, all{arguments.size()};
    for (const TypeAndMode *argument : arguments) {
      withModes += argument->mode.has_value();
    }
    if (result) {
      withModes += result->mode.has_value();
      ++all;
    }
    if (withModes != 0 && withModes != all) {
      state.messages.Say(GetSource(t, state.tokens),
          "either every argument%s or none must have a mode"_err_en_US,
          isFunction ? " and the result" : "");
      return std::nullopt;
    }
    return state.arena.Make<PredDecl>(t.tokens(), layers.kind,
        universalVars, existentialVars, layers.purity, name, arguments,
        result, withType, withInst, det, existentialConstraints,
        constraints);
  }
  void Expected(llvm::raw_ostream &os) const {
    os << "a predicate or function declaration";
  }
};

class ModeDeclRule {
public:
  using resultType = const ModeDecl *;
  bool Matches(const Term &t) const {
    std::optional<DeclarationLayers> layers{PeelDeclaration(t)};
    return layers && layers->isMode;
  }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    DeclarationLayers layers{*PeelDeclaration(t)};
    // A function's mode declaration is the one with a result.
    bool isFunction{AsCompound(
        *SplitBody(*layers.body, false).head, "=", 2) != nullptr};
    DeclarationBody body{SplitBody(*layers.body, isFunction)};
    bool ok{true};
    std::optional<std::tuple<const SymName *, llvm::ArrayRef<Mode>>> head{
        Apply(modeHead, *body.head, state)};
    ok &= head.has_value();
    std::optional<Mode> result;
    std::optional<Inst> withInst;
    std::optional<Determinism> det;
    llvm::ArrayRef<const InstConstraint *> constraints;
    ok &= ConvertOptional(mode, body.result, result, state);
    ok &= ConvertOptional(inst, body.withInst, withInst, state);
    ok &= ConvertOptional(determinism, body.determinism, det, state);
    ok &= ConvertOptional(commaListOf(instConstraint),
        layers.universalConstraints, constraints, state);
    if (body.withType) {
      SayExpected(state, StripParentheses(*body.withType),
          "a mode declaration without a type");
      ok = false;
    }
    if (!ok) {
      return std::nullopt;
    }
    auto [name, arguments]{*head};
    return state.arena.Make<ModeDecl>(t.tokens(),
        isFunction ? PredOrFunc::Function : PredOrFunc::Predicate, name,
        arguments, result, withInst, det, constraints);
  }
  void Expected(llvm::raw_ostream &os) const { os << "a mode declaration"; }
};

//===----------------------------------------------------------------------===//
// Type, inst and mode definitions
//===----------------------------------------------------------------------===//

// The operands of ',' terms, looking through parentheses.
void TypeAttributesRule::Flatten(
    const Term &term, llvm::SmallVectorImpl<const Term *> &into) {
  const Term &t{StripParentheses(term)};
  if (const CompoundTerm * comma{AsCompound(t, ",", 2)}) {
    Flatten(*comma->arguments()[0], into);
    Flatten(*comma->arguments()[1], into);
  } else {
    into.push_back(&t);
  }
}

std::optional<const TypeAttributes *> TypeAttributesRule::Convert(
    const Term &t, ConvertState &state) const {
  const SymName *equality{nullptr}, *comparison{nullptr};
  std::optional<llvm::ArrayRef<const NameAndArity *>> directArgs;
  bool abstractNoncanonical{false};
  bool ok{true};
  llvm::SmallVector<const Term *, 4> attributes;
  Flatten(t, attributes);
  for (const Term *attribute : attributes) {
    llvm::StringRef kind;
    const Term *value{nullptr};
    if (const CompoundTerm * is{AsCompound(*attribute, "is", 2)}) {
      kind = FunctorName(StripParentheses(*is->arguments()[0]));
      value = is->arguments()[1];
    } else if (llvm::isa<NameTerm>(*attribute)) {
      kind = FunctorName(*attribute);
    }
    bool duplicate{false};
    if (value && (kind == "equality" || kind == "comparison")) {
      const SymName *&into{kind == "equality" ? equality : comparison};
      duplicate = into != nullptr;
      auto name{Apply(expect("a predicate name", symName), *value, state)};
      ok &= name.has_value();
      into = name.value_or(into);
    } else if (value && kind == "direct_arg") {
      duplicate = directArgs.has_value();
      directArgs = Apply(listOf(nameAndArity), *value, state);
      ok &= directArgs.has_value();
    } else if (!value && kind == "type_is_abstract_noncanonical") {
      duplicate = abstractNoncanonical;
      abstractNoncanonical = true;
    } else {
      SayExpected(state, *attribute,
          "'equality is', 'comparison is', 'direct_arg is' or "
          "'type_is_abstract_noncanonical'");
      ok = false;
    }
    if (duplicate) {
      state.messages.Say(GetSource(*attribute, state.tokens),
          "the type attribute '%s' is given more than once"_err_en_US,
          kind.str());
      ok = false;
    }
  }
  if (!ok) {
    return std::nullopt;
  }
  return state.arena.Make<TypeAttributes>(t.tokens(), equality, comparison,
      directArgs.value_or(llvm::ArrayRef<const NameAndArity *>{}),
      abstractNoncanonical);
}


// A constructor's argument: Type, or Field :: Type.
class ConstructorArgRule {
public:
  using resultType = const ConstructorArg *;
  bool Matches(const Term &) const { return true; }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    const Term *typeTerm{&t};
    const SymName *field{nullptr};
    bool ok{true};
    if (const CompoundTerm * colons{AsCompound(t, "::", 2)}) {
      auto name{Apply(
          expect("a field name", symName), *colons->arguments()[0], state)};
      ok = name.has_value();
      field = name.value_or(nullptr);
      typeTerm = colons->arguments()[1];
    }
    std::optional<Type> ty{Apply(type, *typeTerm, state)};
    if (!ok || !ty) {
      return std::nullopt;
    }
    return state.arena.Make<ConstructorArg>(t.tokens(), field, *ty);
  }
  void Expected(llvm::raw_ostream &os) const { os << "a type"; }
};

static constexpr ConstructorArgRule constructorArg;

// A constructor: f(Args), a name, [] or [H | T], or a tuple, which is the
// constructor '{}'; some [Vs] and => Constraints may surround it. In braces,
// a term is the constructor it is, not split at ';' (reference manual,
// "Discriminated unions").
class ConstructorRule {
public:
  using resultType = const Constructor *;
  bool Matches(const Term &) const { return true; }
  std::optional<resultType> Convert(
      const Term &term, ConvertState &state) const {
    const Term *t{&StripParentheses(term)};
    bool ok{true};
    llvm::ArrayRef<llvm::StringRef> existentialVars;
    if (const CompoundTerm * some{AsCompound(*t, "some", 2)}) {
      ok &= ConvertOptional(quantifiedVars, some->arguments()[0],
          existentialVars, state);
      t = &StripParentheses(*some->arguments()[1]);
    }
    llvm::ArrayRef<const ClassConstraint *> constraints;
    if (const CompoundTerm * arrow{AsCompound(*t, "=>", 2)}) {
      ok &= ConvertOptional(commaListOf(classConstraint),
          arrow->arguments()[1], constraints, state);
      t = &StripParentheses(*arrow->arguments()[0]);
    }
    // { Ctor } quotes a constructor.
    if (const auto *tuple{llvm::dyn_cast<TupleTerm>(t)};
        tuple && tuple->elements().size() == 1) {
      t = &StripParentheses(*tuple->elements()[0]);
    }
    const SymName *name{nullptr};
    llvm::ArrayRef<const Term *> argumentTerms;
    const Term *cons[2]; // the head and tail of [H | T]
    auto builtinName{[&](llvm::StringRef text) {
      return state.arena.Make<SymName>(
          t->tokens(), llvm::ArrayRef<llvm::StringRef>{}, text);
    }};
    if (const auto *tuple{llvm::dyn_cast<TupleTerm>(t)}) {
      name = builtinName("{}");
      argumentTerms = tuple->elements();
    } else if (const auto *list{llvm::dyn_cast<ListTerm>(t)};
        list && list->elements().empty() && !list->tail()) {
      name = builtinName("[]");
    } else if (list && list->elements().size() == 1 && list->tail()) {
      name = builtinName("[|]");
      cons[0] = list->elements()[0];
      cons[1] = list->tail();
      argumentTerms = cons;
    } else if (llvm::isa<NameTerm>(*t) || llvm::isa<CompoundTerm>(*t)) {
      llvm::SmallVector<const Term *, 4> parts;
      QualifiedParts(*t, parts);
      const Term &last{*parts.back()};
      if (llvm::isa<NameTerm>(last) || llvm::isa<CompoundTerm>(last)) {
        name = MakeSymName(
            *t, llvm::ArrayRef(parts).drop_back(), FunctorName(last), state);
        ok &= name != nullptr;
        if (const auto *compound{llvm::dyn_cast<CompoundTerm>(&last)}) {
          argumentTerms = compound->arguments();
        }
      } else {
        SayExpected(state, last, "a constructor");
        ok = false;
      }
    } else {
      SayExpected(state, *t, "a constructor");
      ok = false;
    }
    std::optional<llvm::ArrayRef<const ConstructorArg *>> arguments{
        detail::ConvertElements(constructorArg, argumentTerms, state)};
    if (!ok || !arguments) {
      return std::nullopt;
    }
    return state.arena.Make<Constructor>(StripParentheses(term).tokens(),
        existentialVars, name, *arguments, constraints);
  }
  void Expected(llvm::raw_ostream &os) const { os << "a constructor"; }
};

// The body of a discriminated union: constructors separated by ';', where
// the last may be followed by "where Attributes", since where binds more
// tightly than ';'.
struct DuBody {
  llvm::SmallVector<const Term *, 8> constructors;
  const Term *attributes{nullptr};
};

static void SplitDuBody(const Term &term, DuBody &body) {
  const Term &t{StripParentheses(term)};
  if (const CompoundTerm * semicolon{AsCompound(t, ";", 2)}) {
    SplitDuBody(*semicolon->arguments()[0], body);
    SplitDuBody(*semicolon->arguments()[1], body);
  } else {
    body.constructors.push_back(&t);
  }
}

static std::optional<std::pair<const Term *, const Term *>> SplitWhere(
    const Term &t) {
  if (const CompoundTerm * where{AsCompound(t, "where", 2)}) {
    return std::pair{where->arguments()[0], where->arguments()[1]};
  }
  return std::nullopt;
}

class TypeDefnRule {
public:
  using resultType = Item;
  bool Matches(const Term &t) const {
    return AsCompound(t, "type", 1) != nullptr;
  }
  std::optional<Item> Convert(const Term &t, ConvertState &state) const {
    const Term &body{StripParentheses(*llvm::cast<CompoundTerm>(t).arguments()[0])};
    if (const CompoundTerm * arrow{AsCompound(body, "--->", 2)}) {
      return ConvertDu(t, *arrow->arguments()[0], *arrow->arguments()[1], state);
    }
    if (const CompoundTerm * equals{AsCompound(body, "==", 2)}) {
      auto head{Apply(typeHead, *equals->arguments()[0], state)};
      std::optional<Type> ty{Apply(type, *equals->arguments()[1], state)};
      if (!head || !ty) {
        return std::nullopt;
      }
      auto [name, parameters]{*head};
      return state.arena.Make<EquivalenceTypeDefn>(
          t.tokens(), name, parameters, *ty);
    }
    const Term *headTerm{&body};
    const TypeAttributes *attributes{nullptr};
    if (auto where{SplitWhere(body)}) {
      headTerm = where->first;
      if (AsCompound(*headTerm, "==", 2)) {
        state.messages.Say(GetSource(t, state.tokens),
            "an equivalence type cannot have 'where' attributes"_err_en_US);
        return std::nullopt;
      }
      auto converted{Apply(typeAttributes, *where->second, state)};
      if (!converted) {
        return std::nullopt;
      }
      attributes = *converted;
    }
    auto head{Apply(expect("a type", typeHead), *headTerm, state)};
    if (!head) {
      return std::nullopt;
    }
    auto [name, parameters]{*head};
    return state.arena.Make<AbstractTypeDecl>(
        t.tokens(), name, parameters, attributes, false);
  }
  void Expected(llvm::raw_ostream &os) const { os << "a type definition"; }

private:
  static std::optional<Item> ConvertDu(const Term &t, const Term &headTerm,
      const Term &bodyTerm, ConvertState &state) {
    bool ok{true};
    const Term *subtype{&headTerm};
    std::optional<Type> supertype;
    if (const CompoundTerm * below{AsCompound(headTerm, "=<", 2)}) {
      subtype = below->arguments()[0];
      supertype = Apply(type, *below->arguments()[1], state);
      ok &= supertype.has_value();
    }
    auto head{Apply(typeHead, *subtype, state)};
    ok &= head.has_value();
    DuBody body;
    SplitDuBody(bodyTerm, body);
    const TypeAttributes *attributes{nullptr};
    if (auto where{SplitWhere(*body.constructors.back())}) {
      body.constructors.back() = where->first;
      auto converted{Apply(typeAttributes, *where->second, state)};
      ok &= converted.has_value();
      attributes = converted.value_or(nullptr);
    }
    auto constructors{detail::ConvertElements(ConstructorRule{},
        llvm::ArrayRef<const Term *>{body.constructors}, state)};
    if (!ok || !constructors) {
      return std::nullopt;
    }
    auto [name, parameters]{*head};
    return state.arena.Make<DuTypeDefn>(
        t.tokens(), name, parameters, supertype, *constructors, attributes);
  }
};

// :- inst Head == Inst. and :- inst Head ---> Functors., where Head may be
// followed by "for Name/Arity".
class InstDefnRule {
public:
  using resultType = const InstDefn *;
  bool Matches(const Term &t) const {
    const CompoundTerm *decl{AsCompound(t, "inst", 1)};
    return decl &&
        (AsCompound(*decl->arguments()[0], "==", 2) ||
            AsCompound(*decl->arguments()[0], "--->", 2));
  }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    const auto &body{llvm::cast<CompoundTerm>(
        StripParentheses(*llvm::cast<CompoundTerm>(t).arguments()[0]))};
    const Term *headTerm{body.arguments()[0]};
    bool ok{true};
    const NameAndArity *forType{nullptr};
    if (const CompoundTerm * forTerm{AsCompound(*headTerm, "for", 2)}) {
      headTerm = forTerm->arguments()[0];
      ok &= ConvertOptional(nameAndArity, forTerm->arguments()[1], forType,
          state);
    }
    auto head{Apply(instHead, *headTerm, state)};
    ok &= head.has_value();
    std::optional<Inst> definition;
    if (FunctorName(body) == "==") {
      definition = Apply(inst, *body.arguments()[1], state);
    } else {
      definition = ConvertFunctors(*body.arguments()[1], state);
    }
    if (!ok || !definition) {
      return std::nullopt;
    }
    auto [name, parameters]{*head};
    return state.arena.Make<InstDefn>(
        t.tokens(), name, parameters, forType, *definition);
  }
  void Expected(llvm::raw_ostream &os) const { os << "an inst definition"; }

private:
  // Functors ---> F1 ; F2 means bound(F1 ; F2).
  static std::optional<Inst> ConvertFunctors(
      const Term &t, ConvertState &state) {
    llvm::SmallVector<const BoundFunctor *, 8> functors;
    bool ok{true};
    auto collect{[&](const Term &term, auto &self) -> void {
      if (const CompoundTerm * semicolon{AsCompound(term, ";", 2)}) {
        for (const Term *side : semicolon->arguments()) {
          self(*side, self);
        }
      } else if (auto functor{Apply(BoundFunctorRule{}, term, state)}) {
        functors.push_back(*functor);
      } else {
        ok = false;
      }
    }};
    collect(t, collect);
    if (!ok) {
      return std::nullopt;
    }
    return state.arena.Make<BoundInst>(StripParentheses(t).tokens(), false,
        state.arena.SaveArray(llvm::ArrayRef<const BoundFunctor *>{functors}));
  }
};

// :- mode Head == Mode.
class ModeDefnRule {
public:
  using resultType = const ModeDefn *;
  bool Matches(const Term &t) const {
    const CompoundTerm *decl{AsCompound(t, "mode", 1)};
    return decl && AsCompound(*decl->arguments()[0], "==", 2);
  }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    const CompoundTerm &body{*AsCompound(
        *llvm::cast<CompoundTerm>(t).arguments()[0], "==", 2)};
    auto head{Apply(instHead, *body.arguments()[0], state)};
    std::optional<Mode> definition{Apply(mode, *body.arguments()[1], state)};
    if (!head || !definition) {
      return std::nullopt;
    }
    auto [name, parameters]{*head};
    return state.arena.Make<ModeDefn>(
        t.tokens(), name, parameters, *definition);
  }
  void Expected(llvm::raw_ostream &os) const { os << "a mode definition"; }
};

//===----------------------------------------------------------------------===//
// Type classes
//===----------------------------------------------------------------------===//

// The variables of one side of a functional dependency: A, or A, B.
static constexpr auto dependencyVars{
    commaListOf(expect("a type variable", variable))};

static constexpr auto functionalDependency{construct<FunctionalDependency>(
    functor("->", dependencyVars, dependencyVars))};

static constexpr auto typeclassConstraints{commaListOf(expect(
    "a superclass constraint or a functional dependency such as '(A -> B)'",
    anyOf<TypeclassConstraint>(functionalDependency, classConstraint)))};

// A method is a pred, func or mode declaration without ':-'.
static constexpr auto classMethod{expect("a pred, func or mode declaration",
    anyOf<ClassMethod>(PredDeclRule{}, ModeDeclRule{}))};

static constexpr auto methodList{
    expect("a list of methods", listOf(classMethod))};

// The parts of a typeclass or instance declaration: Head <= Constraints
// where [Methods], where the constraints and the methods are optional.
struct ClassParts {
  const Term *head;
  const Term *constraints{nullptr};
  const Term *methods{nullptr};
};

static ClassParts SplitClass(const Term &term) {
  ClassParts parts{&StripParentheses(term)};
  if (auto where{SplitWhere(*parts.head)}) {
    parts.head = &StripParentheses(*where->first);
    parts.methods = where->second;
  }
  if (const CompoundTerm * constrained{AsCompound(*parts.head, "<=", 2)}) {
    parts.head = constrained->arguments()[0];
    parts.constraints = constrained->arguments()[1];
  }
  return parts;
}

class TypeclassDeclRule {
public:
  using resultType = const TypeclassDecl *;
  bool Matches(const Term &t) const { return AsCompound(t, "typeclass", 1); }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    ClassParts parts{SplitClass(*llvm::cast<CompoundTerm>(t).arguments()[0])};
    bool ok{true};
    auto head{Apply(expect("a type class name and its parameters",
                        distinctParameters(appliedName(
                            expect("a type class parameter", variable)))),
        *parts.head, state)};
    ok &= head.has_value();
    llvm::ArrayRef<TypeclassConstraint> constraints;
    llvm::ArrayRef<ClassMethod> methods;
    ok &= ConvertOptional(
        typeclassConstraints, parts.constraints, constraints, state);
    ok &= ConvertOptional(methodList, parts.methods, methods, state);
    if (!ok) {
      return std::nullopt;
    }
    auto [name, parameters]{*head};
    if (parameters.empty()) {
      state.messages.Say(GetSource(*parts.head, state.tokens),
          "a type class must have at least one parameter"_err_en_US);
      return std::nullopt;
    }
    // As in mmc, a functional dependency relates the class's parameters.
    for (const TypeclassConstraint &constraint : constraints) {
      if (const auto *dependency{
              std::get_if<const FunctionalDependency *>(&constraint)}) {
        for (auto vars : {(*dependency)->domain, (*dependency)->range}) {
          for (llvm::StringRef var : vars) {
            if (!llvm::is_contained(parameters, var)) {
              state.messages.Say(GetSource(*parts.constraints, state.tokens),
                  "the variable %s in a functional dependency is not a "
                  "parameter of the type class"_err_en_US,
                  var.str());
              ok = false;
            }
          }
        }
      }
    }
    if (!ok) {
      return std::nullopt;
    }
    return state.arena.Make<TypeclassDecl>(t.tokens(), name, parameters,
        constraints, parts.methods == nullptr, methods);
  }
  void Expected(llvm::raw_ostream &os) const {
    os << "a typeclass declaration";
  }
};

// pred(Name/Arity) is Implementation or func(Name/Arity) is Implementation,
// or a clause.
class InstanceMethodRule {
public:
  using resultType = InstanceMethod;
  bool Matches(const Term &) const { return true; }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    const CompoundTerm *is{AsCompound(t, "is", 2)};
    const CompoundTerm *kind{nullptr};
    if (is) {
      kind = AsCompound(*is->arguments()[0], "pred", 1);
      if (!kind) {
        kind = AsCompound(*is->arguments()[0], "func", 1);
      }
    }
    if (!kind) {
      if (std::optional<Clause> converted{clause.Convert(t, state)}) {
        return std::visit(
            [](auto *node) -> InstanceMethod { return node; }, *converted);
      }
      return std::nullopt;
    }
    auto method{Apply(nameAndArity, *kind->arguments()[0], state)};
    auto implementation{Apply(
        expect("a predicate or function name", symName), *is->arguments()[1],
        state)};
    if (!method || !implementation) {
      return std::nullopt;
    }
    return state.arena.Make<MethodByName>(t.tokens(),
        FunctorName(*kind) == "func" ? PredOrFunc::Function
                                     : PredOrFunc::Predicate,
        *method, *implementation);
  }
  void Expected(llvm::raw_ostream &os) const { os << "an instance method"; }
};

class InstanceDeclRule {
public:
  using resultType = const InstanceDecl *;
  bool Matches(const Term &t) const { return AsCompound(t, "instance", 1); }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    ClassParts parts{SplitClass(*llvm::cast<CompoundTerm>(t).arguments()[0])};
    bool ok{true};
    auto head{Apply(expect("a type class name and its types",
                        appliedName(type)),
        *parts.head, state)};
    ok &= head.has_value();
    llvm::ArrayRef<const ClassConstraint *> constraints;
    llvm::ArrayRef<InstanceMethod> methods;
    ok &= ConvertOptional(commaListOf(classConstraint), parts.constraints,
        constraints, state);
    ok &= ConvertOptional(
        expect("a list of methods", listOf(InstanceMethodRule{})),
        parts.methods, methods, state);
    if (!ok) {
      return std::nullopt;
    }
    auto [name, types]{*head};
    return state.arena.Make<InstanceDecl>(t.tokens(), name, types,
        constraints, parts.methods == nullptr, methods);
  }
  void Expected(llvm::raw_ostream &os) const {
    os << "an instance declaration";
  }
};

//===----------------------------------------------------------------------===//
// Items (reference manual, "Items")
//===----------------------------------------------------------------------===//

static constexpr auto declaration{anyOf<Item>(moduleDecl, endModuleDecl,
    sectionDecl, importDecl, PredDeclRule{}, ModeDeclRule{}, TypeDefnRule{},
    InstDefnRule{}, ModeDefnRule{}, TypeclassDeclRule{}, InstanceDeclRule{},
    pragma, InitialiseRule{}, MutableRule{}, PromiseRule{},
    PromiseDisjunctionRule{}, SolverTypeRule{})};

std::optional<Item> ConvertItem(const Term &term, ConvertState &state) {
  const Term &t{StripParentheses(term)};
  if (const auto *directive{llvm::dyn_cast<DirectiveTerm>(&t)}) {
    return ConvertItem(IgnoreDirective(*directive, state), state);
  }
  if (!(llvm::isa<CompoundTerm>(t) && FunctorName(t) == ":-" &&
          Arity(t) == 1)) {
    // A rule, a DCG rule or a fact.
    if (std::optional<Clause> converted{clause.Convert(t, state)}) {
      return std::visit([](auto *node) -> Item { return node; }, *converted);
    }
    return std::nullopt;
  }
  const Term &decl{
      StripParentheses(*llvm::cast<CompoundTerm>(t).arguments()[0])};
  if (declaration.Matches(decl)) {
    return declaration.Convert(decl, state);
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
