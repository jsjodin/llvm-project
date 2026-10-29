//===-- lib/Parser/convert-rules.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_LIB_PARSER_CONVERT_RULES_H_
#define MERKURIUS_LIB_PARSER_CONVERT_RULES_H_

// The converter's rules that its files share: convert.cpp, for
// declarations, convert-goals.cpp, for clauses, goals and data-terms, and
// convert-pragmas.cpp, for pragmas.

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/term-combinators.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include <optional>
#include <string>
#include <utility>

namespace merkurius::parser {

// The first line of a term's source, shortened to fit in a message.
std::string Excerpt(const Term &, const TokenSequence &);

// Warns that a %$omp directive is not before a goal, where it would apply,
// and returns the term it annotates, which is converted without it.
inline const Term &IgnoreDirective(
    const DirectiveTerm &directive, ConvertState &state) {
  state.messages.Say(
      state.tokens.tokens.at(directive.directiveToken()).spelling,
      "an OpenMP directive applies only to a goal; it is ignored"_warn_en_US);
  return *directive.annotated();
}

// The compound term `t` with this functor and arity, if it is one.
inline const CompoundTerm *AsCompound(
    const Term &t, llvm::StringRef name, std::size_t arity) {
  const auto *compound{llvm::dyn_cast<CompoundTerm>(&StripParentheses(t))};
  return compound && FunctorName(*compound) == name &&
          Arity(*compound) == arity
      ? compound
      : nullptr;
}

// The operands of name(X, Y), written as an operator term or as X `name` Y.
inline std::optional<std::pair<const Term *, const Term *>> BinaryOperands(
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

// Converts the optional part `term` with `rule` into `into`, or returns
// false after an error.
template <typename R, typename T>
bool ConvertOptional(
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

// The categories that contain themselves are classes, declared here and
// defined in convert.cpp after the rules that refer to them.
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

inline constexpr TypeRule type;
inline constexpr InstRule inst;
inline constexpr ModeRule mode;
inline constexpr TypeAndModeRule typeAndMode;

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
  // The category that `name` names, as in require_det.
  static std::optional<Determinism> FromName(llvm::StringRef name) {
    return llvm::StringSwitch<std::optional<Determinism>>(name)
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

private:
  static std::optional<Determinism> Lookup(const Term &t) {
    if (!llvm::isa<NameTerm>(t)) {
      return std::nullopt;
    }
    return FromName(FunctorName(t));
  }
};

inline constexpr DeterminismRule determinism;

// An integer, float or string. An integer must fit its type (reference
// manual, "Builtin types"); the reader reads one of any size, as mmc does.
class LiteralRule {
public:
  using resultType = const Literal *;
  bool Matches(const Term &t) const {
    return llvm::isa<IntegerTerm>(t) || llvm::isa<FloatTerm>(t) ||
        llvm::isa<StringTerm>(t);
  }
  std::optional<resultType> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a literal"; }
};

inline constexpr LiteralRule literal;

inline constexpr auto quantifiedVars{
    expect("a list of variables", listOf(variable))};


// Name/Arity, with a small nonnegative integer arity.
class NameAndArityRule {
public:
  using resultType = const NameAndArity *;
  bool Matches(const Term &t) const { return AsCompound(t, "/", 2); }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    llvm::ArrayRef<const Term *> parts{llvm::cast<CompoundTerm>(t).arguments()};
    std::optional<const SymName *> name{Apply(symName, *parts[0], state)};
    const Term &arity{StripParentheses(*parts[1])};
    const auto *integer{llvm::dyn_cast<IntegerTerm>(&arity)};
    if (!integer || integer->negative() ||
        integer->size() != IntegerSize::Default ||
        integer->magnitude().getActiveBits() > 16) {
      SayExpected(state, arity, "an arity");
      return std::nullopt;
    }
    if (!name) {
      return std::nullopt;
    }
    return state.arena.Make<NameAndArity>(t.tokens(), *name,
        static_cast<unsigned>(integer->magnitude().getZExtValue()));
  }
  void Expected(llvm::raw_ostream &os) const { os << "a name/arity"; }
};

inline constexpr NameAndArityRule nameAndArity;

// The name being defined and its parameters, which are variables: t(T1, T2).
inline constexpr auto typeHeadRule{
    appliedName(expect("a type parameter", variable))};
inline constexpr auto instHeadRule{
    appliedName(expect("an inst parameter", variable))};

// A head whose parameters are distinct.
template <typename R> class DefinitionHeadRule {
public:
  using resultType = typename R::resultType;
  constexpr explicit DefinitionHeadRule(R rule) : rule_{rule} {}
  bool Matches(const Term &t) const { return rule_.Matches(t); }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    std::optional<resultType> head{rule_.Convert(t, state)};
    if (!head) {
      return std::nullopt;
    }
    llvm::ArrayRef<llvm::StringRef> parameters{std::get<1>(*head)};
    for (std::size_t j{0}; j < parameters.size(); ++j) {
      if (llvm::is_contained(parameters.take_front(j), parameters[j])) {
        state.messages.Say(GetSource(t, state.tokens),
            "the parameter %s appears more than once"_err_en_US,
            parameters[j].str());
        return std::nullopt;
      }
    }
    return head;
  }
  void Expected(llvm::raw_ostream &os) const { rule_.Expected(os); }

private:
  R rule_;
};

template <typename R>
constexpr DefinitionHeadRule<R> distinctParameters(R rule) {
  return DefinitionHeadRule<R>{rule};
}

inline constexpr auto typeHead{distinctParameters(typeHeadRule)};
inline constexpr auto instHead{distinctParameters(instHeadRule)};

// The attributes after where, each given at most once.
class TypeAttributesRule {
public:
  using resultType = const TypeAttributes *;
  bool Matches(const Term &) const { return true; }
  std::optional<resultType> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "type attributes"; }

private:
  static void Flatten(const Term &, llvm::SmallVectorImpl<const Term *> &);
};

inline constexpr TypeAttributesRule typeAttributes;

// A clause: a fact, a rule or a DCG rule, of a predicate or a function
// (reference manual, "Facts", "Rules" and "DCG-rules").
class ClauseRule {
public:
  using resultType = Clause;
  bool Matches(const Term &) const { return true; }
  std::optional<Clause> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a clause"; }
};

inline constexpr ClauseRule clause;

// :- pragma Name(Arguments), without the ':-'.
class PragmaRule {
public:
  using resultType = Item;
  bool Matches(const Term &t) const { return AsCompound(t, "pragma", 1); }
  std::optional<Item> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a pragma"; }
};

inline constexpr PragmaRule pragma;

} // namespace merkurius::parser
#endif // MERKURIUS_LIB_PARSER_CONVERT_RULES_H_
