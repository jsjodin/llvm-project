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

// The categories that contain themselves are classes, declared here and
// defined after the rules that refer to them.
struct TypeRule {
  using resultType = Type;
  bool Matches(const Term &) const { return true; }
  std::optional<Type> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a type"; }
};
struct InstRule {
  using resultType = Inst;
  bool Matches(const Term &) const { return true; }
  std::optional<Inst> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "an inst"; }
};
struct ModeRule {
  using resultType = Mode;
  bool Matches(const Term &) const { return true; }
  std::optional<Mode> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a mode"; }
};
// Type, or Type::Mode.
struct TypeAndModeRule {
  using resultType = const TypeAndMode *;
  bool Matches(const Term &) const { return true; }
  std::optional<resultType> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a type"; }
};

static constexpr TypeRule type;
static constexpr InstRule inst;
static constexpr ModeRule mode;
static constexpr TypeAndModeRule typeAndMode;

// The compound term `t` with this functor and arity, if it is one.
static const CompoundTerm *AsCompound(
    const Term &t, llvm::StringRef name, std::size_t arity) {
  const auto *compound{llvm::dyn_cast<CompoundTerm>(&StripParentheses(t))};
  return compound && FunctorName(*compound) == name &&
          Arity(*compound) == arity
      ? compound
      : nullptr;
}

// The determinism categories (reference manual, "Determinism categories").
class DeterminismRule {
public:
  using resultType = Determinism;
  bool Matches(const Term &t) const { return Lookup(t).has_value(); }
  std::optional<Determinism> Convert(const Term &t, ConvertState &) const {
    return Lookup(t);
  }
  void Expected(llvm::raw_ostream &os) const {
    os << "a determinism category";
  }

private:
  static std::optional<Determinism> Lookup(const Term &t) {
    if (!llvm::isa<NameTerm>(t)) {
      return std::nullopt;
    }
    return llvm::StringSwitch<std::optional<Determinism>>(FunctorName(t))
        .Case("det", Determinism::Det)
        .Case("semidet", Determinism::Semidet)
        .Case("multi", Determinism::Multi)
        .Case("nondet", Determinism::Nondet)
        .Case("failure", Determinism::Failure)
        .Case("erroneous", Determinism::Erroneous)
        .Case("cc_multi", Determinism::CcMulti)
        .Case("cc_nondet", Determinism::CcNondet)
        .Default(std::nullopt);
  }
};

static constexpr DeterminismRule determinism;

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

// A function symbol of a bound inst: f(I1, I2), a, [] or [I | J].
class BoundFunctorRule {
public:
  using resultType = const BoundFunctor *;
  bool Matches(const Term &t) const {
    if (const auto *list{llvm::dyn_cast<ListTerm>(&t)}) {
      return list->elements().empty() ||
          (list->elements().size() == 1 && list->tail());
    }
    return functor_.Matches(t);
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

static constexpr auto quantifiedVars{
    expect("a list of variables", listOf(variable))};

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

// The operands of name(X, Y), written as an operator term or as X `name` Y.
static std::optional<std::pair<const Term *, const Term *>> BinaryOperands(
    const Term &term, llvm::StringRef name) {
  const Term &t{StripParentheses(term)};
  if (const CompoundTerm * c{AsCompound(t, name, 2)}) {
    return std::pair{c->arguments()[0], c->arguments()[1]};
  }
  if (const auto *backquote{llvm::dyn_cast<BackquoteTerm>(&t)}) {
    const Term &op{StripParentheses(*backquote->op())};
    if (llvm::isa<NameTerm>(op) && FunctorName(op) == name) {
      return std::pair{backquote->left(), backquote->right()};
    }
  }
  return std::nullopt;
}

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

// Converts the optional part `term` with `rule` into `into`, or returns
// false after an error.
template <typename R, typename T>
static bool ConvertOptional(
    const R &rule, const Term *term, T &into, ConvertState &state) {
  if (!term) {
    return true;
  }
  if (auto result{Apply(rule, *term, state)}) {
    into = *result;
    return true;
  }
  return false;
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
// Items (reference manual, "Items")
//===----------------------------------------------------------------------===//

static constexpr auto declaration{anyOf<Item>(moduleDecl, endModuleDecl,
    sectionDecl, importDecl, PredDeclRule{}, ModeDeclRule{})};

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
