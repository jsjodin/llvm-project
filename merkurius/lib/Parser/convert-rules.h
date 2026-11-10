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
// declarations, convert-goals.cpp, for clauses, goals and data-terms,
// convert-pragmas.cpp, for pragmas, convert-directives.cpp, for
// initialise, finalise, mutable, promises and solver types, and
// convert-omp.cpp, for %$omp directives.

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

// A goal with a %$omp directive, defined in convert-omp.cpp.
std::optional<Goal> ConvertOmpGoal(const DirectiveTerm &, ConvertState &);

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

// A function's head: a name and its arguments converted with P or, at the
// top level of a function declaration, mode declaration or clause, field
// access syntax (reference manual, "User-supplied field access function
// declarations"). T ^ F(A1, ...) is F(A1, ..., T), and (T ^ F(A1, ...) :=
// V) is 'F :='(A1, ..., T, V). As in mmc, the field must be a name or a
// compound term, so a clause X ^ Y = Z defines '^'/2.
template <typename P> class FunctionHeadRule {
public:
  using resultType = typename AppliedNameRule<P>::resultType;
  constexpr explicit FunctionHeadRule(P rule) : rule_{rule} {}
  bool Matches(const Term &t) const { return appliedName(rule_).Matches(t); }
  std::optional<resultType> Convert(
      const Term &term, ConvertState &state) const {
    const Term &t{StripParentheses(term)};
    const CompoundTerm *update{AsCompound(t, ":=", 2)};
    const CompoundTerm *select{
        AsCompound(update ? *update->arguments()[0] : t, "^", 2)};
    llvm::SmallVector<const Term *, 4> parts;
    if (select) {
      QualifiedParts(StripParentheses(*select->arguments()[1]), parts);
    }
    if (!select ||
        !(llvm::isa<NameTerm>(*parts.back()) ||
            llvm::isa<CompoundTerm>(*parts.back()))) {
      return appliedName(rule_).Convert(term, state);
    }
    const Term &field{StripParentheses(*select->arguments()[1])};
    const Term &part{*parts.back()};
    const SymName *name{
        MakeSymName(field, llvm::ArrayRef(parts).drop_back(), part, state)};
    if (!name) {
      return std::nullopt;
    }
    if (update) {
      name = state.arena.Make<SymName>(name->tokens, name->qualifiers,
          state.arena.Save((name->name + " :=").str()));
    }
    llvm::SmallVector<const Term *, 4> terms;
    if (const auto *compound{llvm::dyn_cast<CompoundTerm>(&part)}) {
      llvm::append_range(terms, compound->arguments());
    }
    terms.push_back(select->arguments()[0]);
    if (update) {
      terms.push_back(update->arguments()[1]);
    }
    auto arguments{detail::ConvertElements(
        rule_, llvm::ArrayRef<const Term *>{terms}, state)};
    if (!arguments) {
      return std::nullopt;
    }
    return resultType{name, *arguments};
  }
  void Expected(llvm::raw_ostream &os) const { os << "a name"; }

private:
  P rule_;
};

template <typename P> constexpr FunctionHeadRule<P> functionHead(P rule) {
  return FunctionHeadRule<P>{rule};
}

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

// Data-terms and goals, defined in convert-goals.cpp.
struct DataTermRule {
  using resultType = DataTerm;
  bool Matches(const Term &) const { return true; }
  std::optional<DataTerm> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a data-term"; }
};
struct GoalRule {
  using resultType = Goal;
  bool Matches(const Term &) const { return true; }
  std::optional<Goal> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a goal"; }
};

inline constexpr DataTermRule dataTerm;
inline constexpr GoalRule goal;

// A string literal's characters.
struct StringRule {
  using resultType = llvm::StringRef;
  bool Matches(const Term &t) const { return llvm::isa<StringTerm>(t); }
  std::optional<llvm::StringRef> Convert(
      const Term &t, ConvertState &state) const {
    return state.arena.Save(llvm::cast<StringTerm>(t).value());
  }
  void Expected(llvm::raw_ostream &os) const { os << "a string"; }
};

inline constexpr StringRule stringLiteral;

// "C", "C#" or "Java", or c, csharp or java (mmc: "The valid languages are
// c, C, csharp, C#, java and Java").
struct LanguageRule {
  using resultType = ForeignLanguage;
  bool Matches(const Term &t) const { return Lookup(t).has_value(); }
  std::optional<ForeignLanguage> Convert(const Term &t, ConvertState &) const {
    return Lookup(t);
  }
  void Expected(llvm::raw_ostream &os) const {
    os << "a foreign language: \"C\", \"C#\" or \"Java\"";
  }

private:
  static std::optional<ForeignLanguage> Lookup(const Term &t) {
    llvm::StringRef name;
    if (const auto *s{llvm::dyn_cast<StringTerm>(&t)}) {
      name = s->value();
    } else if (llvm::isa<NameTerm>(t)) {
      name = FunctorName(t);
    } else {
      return std::nullopt;
    }
    return llvm::StringSwitch<std::optional<ForeignLanguage>>(name)
        .Cases({"c", "C"}, ForeignLanguage::C)
        .Cases({"csharp", "C#"}, ForeignLanguage::CSharp)
        .Cases({"java", "Java"}, ForeignLanguage::Java)
        .Default(std::nullopt);
  }
};

inline constexpr LanguageRule language;

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

// The declarations of convert-directives.cpp, without the ':-'.

// mutable(Name, Type, Initial, Inst, [Attributes]).
struct MutableRule {
  using resultType = const MutableDecl *;
  bool Matches(const Term &t) const {
    return llvm::isa<CompoundTerm>(t) && FunctorName(t) == "mutable";
  }
  std::optional<resultType> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a mutable declaration"; }
};

// process Kind receives Types [on gpu], with -fprocess-declarations
// (convert-processes.cpp).
struct ProcessRule {
  using resultType = const ProcessDecl *;
  bool Matches(const Term &t) const { return AsCompound(t, "process", 1); }
  std::optional<resultType> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a process declaration"; }
};

// initialise p/2 and finalise p/2, with their spellings initialize and
// finalize.
struct InitialiseRule {
  using resultType = Item;
  bool Matches(const Term &t) const {
    return llvm::isa<CompoundTerm>(t) && Arity(t) == 1 &&
        llvm::is_contained(
            {"initialise", "initialize", "finalise", "finalize"},
            FunctorName(t));
  }
  std::optional<Item> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const {
    os << "an initialise or finalise declaration";
  }
};

// promise Goal.
struct PromiseRule {
  using resultType = Item;
  bool Matches(const Term &t) const { return AsCompound(t, "promise", 1); }
  std::optional<Item> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a promise"; }
};

// all [Vars] promise_exclusive Disjunction, and promise_exhaustive and
// promise_exclusive_exhaustive.
struct PromiseDisjunctionRule {
  using resultType = Item;
  bool Matches(const Term &) const;
  std::optional<Item> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const {
    os << "a promise about a disjunction";
  }

private:
  static std::optional<PromiseDisjunctionKind> Kind(const Term &);
};

// solver type t, or solver type t where Attributes.
struct SolverTypeRule {
  using resultType = Item;
  bool Matches(const Term &t) const { return AsCompound(t, "solver", 1); }
  std::optional<Item> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a solver type"; }
};

} // namespace merkurius::parser
#endif // MERKURIUS_LIB_PARSER_CONVERT_RULES_H_
