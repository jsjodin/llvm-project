//===-- lib/Parser/convert-omp.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The converter's rules for %$omp directives (design document, "OpenMP
// directives"). A directive is read from its own tokens: its name, the
// longest run of names that OMP.td knows as a directive, in its Fortran
// spelling; the directive's arguments, if its last name is applied to
// some; and its clauses, each a name, perhaps applied to arguments, and
// perhaps separated by commas. Arguments are read by the term reader and
// converted as data-terms. As in Flang, the clauses with structure that
// later passes use have nodes of their own; the others keep their
// arguments as data-terms.

#include "convert-rules.h"
#include "merkurius/Parser/term-reader.h"
#include "llvm/ADT/StringMap.h"

namespace merkurius::parser {

namespace {

// The numbers of the named variables in a term.
void NumberVariables(const Term &t, llvm::StringMap<unsigned> &numbers) {
  auto all{[&](llvm::ArrayRef<const Term *> terms) {
    for (const Term *term : terms) {
      if (term) {
        NumberVariables(*term, numbers);
      }
    }
  }};
  if (const auto *variable{llvm::dyn_cast<VariableTerm>(&t)}) {
    if (!variable->IsAnonymous()) {
      numbers.try_emplace(variable->name(), variable->number());
    }
  } else if (const auto *compound{llvm::dyn_cast<CompoundTerm>(&t)}) {
    all(compound->arguments());
  } else if (const auto *backquote{llvm::dyn_cast<BackquoteTerm>(&t)}) {
    all({backquote->left(), backquote->op(), backquote->right()});
  } else if (const auto *list{llvm::dyn_cast<ListTerm>(&t)}) {
    all(list->elements());
    all({list->tail()});
  } else if (const auto *tuple{llvm::dyn_cast<TupleTerm>(&t)}) {
    all(tuple->elements());
  } else if (const auto *apply{llvm::dyn_cast<ApplyTerm>(&t)}) {
    all({apply->closure()});
    all(apply->arguments());
  } else if (const auto *parenthesized{
                 llvm::dyn_cast<ParenthesizedTerm>(&t)}) {
    NumberVariables(*parenthesized->inner(), numbers);
  } else if (const auto *directive{llvm::dyn_cast<DirectiveTerm>(&t)}) {
    NumberVariables(*directive->annotated(), numbers);
  }
}

// A clause's arguments, split at the first one written M : X: the modifiers
// are the arguments before it and M, and the rest are X and those after.
struct Arguments {
  llvm::SmallVector<const Term *, 4> modifiers;
  llvm::SmallVector<const Term *, 4> rest;
  bool colon{false};
};

Arguments SplitArguments(llvm::ArrayRef<const Term *> arguments) {
  Arguments split;
  for (const Term *argument : arguments) {
    const auto *compound{llvm::dyn_cast<CompoundTerm>(argument)};
    if (!split.colon && compound && FunctorName(*compound) == ":" &&
        Arity(*compound) == 2 && compound->form() == CompoundForm::Infix) {
      split.modifiers.append(split.rest.begin(), split.rest.end());
      split.rest.clear();
      split.modifiers.push_back(compound->arguments()[0]);
      split.rest.push_back(compound->arguments()[1]);
      split.colon = true;
    } else {
      split.rest.push_back(argument);
    }
  }
  return split;
}

template <typename E> using Keywords = std::pair<llvm::StringLiteral, E>;

// The enum value a keyword such as dynamic names.
template <typename E> class KeywordRule {
public:
  using resultType = E;
  constexpr KeywordRule(
      llvm::StringLiteral what, llvm::ArrayRef<Keywords<E>> keywords)
      : what_{what}, keywords_{keywords} {}
  bool Matches(const Term &t) const { return Find(t).has_value(); }
  std::optional<E> Convert(const Term &t, ConvertState &) const {
    return Find(t);
  }
  void Expected(llvm::raw_ostream &os) const { os << what_; }

private:
  std::optional<E> Find(const Term &t) const {
    if (llvm::isa<NameTerm>(t)) {
      for (const auto &[name, value] : keywords_) {
        if (FunctorName(t) == name) {
          return value;
        }
      }
    }
    return std::nullopt;
  }
  llvm::StringLiteral what_;
  llvm::ArrayRef<Keywords<E>> keywords_;
};

constexpr Keywords<OmpScheduleModifier> scheduleModifiers[]{
    {"monotonic", OmpScheduleModifier::Monotonic},
    {"nonmonotonic", OmpScheduleModifier::Nonmonotonic},
    {"simd", OmpScheduleModifier::Simd}};
constexpr Keywords<OmpScheduleKind> scheduleKinds[]{
    {"static", OmpScheduleKind::Static}, {"dynamic", OmpScheduleKind::Dynamic},
    {"guided", OmpScheduleKind::Guided}, {"auto", OmpScheduleKind::Auto},
    {"runtime", OmpScheduleKind::Runtime}};
constexpr Keywords<OmpReductionModifier> reductionModifiers[]{
    {"default", OmpReductionModifier::Default},
    {"inscan", OmpReductionModifier::Inscan},
    {"task", OmpReductionModifier::Task}};
constexpr Keywords<OmpMapModifier> mapModifiers[]{
    {"always", OmpMapModifier::Always}, {"close", OmpMapModifier::Close},
    {"present", OmpMapModifier::Present}};
constexpr Keywords<OmpMapType> mapTypes[]{{"alloc", OmpMapType::Alloc},
    {"delete", OmpMapType::Delete}, {"from", OmpMapType::From},
    {"release", OmpMapType::Release}, {"storage", OmpMapType::Storage},
    {"to", OmpMapType::To}, {"tofrom", OmpMapType::Tofrom}};

const KeywordRule<OmpScheduleModifier> scheduleModifier{
    "a schedule modifier: monotonic, nonmonotonic or simd", scheduleModifiers};
const KeywordRule<OmpScheduleKind> scheduleKind{
    "a schedule kind: static, dynamic, guided, auto or runtime",
    scheduleKinds};
const KeywordRule<OmpReductionModifier> reductionModifier{
    "a reduction modifier: default, inscan or task", reductionModifiers};
const KeywordRule<OmpMapModifier> mapModifier{
    "a map modifier: always, close or present", mapModifiers};
const KeywordRule<OmpMapType> mapType{
    "a map type: alloc, delete, from, release, storage, to or tofrom",
    mapTypes};

// A variable, as a parse-tree node.
class VariableNodeRule {
public:
  using resultType = const Variable *;
  bool Matches(const Term &t) const { return llvm::isa<VariableTerm>(t); }
  std::optional<const Variable *> Convert(
      const Term &t, ConvertState &state) const {
    const auto &variable{llvm::cast<VariableTerm>(t)};
    return state.arena.Make<Variable>(
        t.tokens(), state.arena.Save(variable.name()), variable.number());
  }
  void Expected(llvm::raw_ostream &os) const { os << "a variable"; }
};

inline constexpr VariableNodeRule variableNode;

class DirectiveConverter {
public:
  DirectiveConverter(std::uint32_t directive, ConvertState &outer)
      : outer_{outer}, directive_{directive},
        tokens_{GetDirectiveTokens(outer.tokens, directive)},
        state_{tokens_, outer.arena, outer.messages},
        reader_{tokens_, terms_, outer.messages} {
    if (outer.item) {
      NumberVariables(*outer.item, numbers_);
    }
  }

  const OmpDirective *Convert();

private:
  const Token &At(std::uint32_t j) const {
    return tokens_.tokens[std::min<std::size_t>(j, tokens_.tokens.size() - 1)];
  }
  TokenKind KindAt(std::uint32_t j) const { return At(j).kind; }
  // The name at `at`, applied to arguments if they follow it, which are
  // put in `arguments`. Moves `at` past it.
  bool ReadApplied(std::uint32_t &at,
      llvm::SmallVectorImpl<const Term *> &arguments);
  std::optional<OmpClause> Clause(std::uint32_t &at);
  template <typename R>
  std::optional<llvm::ArrayRef<typename R::resultType>> Elements(
      const R &rule, llvm::ArrayRef<const Term *> terms) {
    return detail::ConvertElements(rule, terms, state_);
  }

  ConvertState &outer_;
  std::uint32_t directive_;
  TokenSequence tokens_;
  ConvertState state_;
  TermArena terms_;
  TermReader reader_;
  llvm::StringMap<unsigned> numbers_;
};

bool DirectiveConverter::ReadApplied(
    std::uint32_t &at, llvm::SmallVectorImpl<const Term *> &arguments) {
  if (KindAt(at + 1) != TokenKind::OpenCt) {
    ++at;
    return true;
  }
  unsigned count{outer_.variables};
  for (const auto &entry : numbers_) {
    count = std::max(count, entry.second + 1);
  }
  const Term *term{reader_.ReadApplication(at, numbers_, count)};
  if (!term) {
    return false;
  }
  const auto *applied{llvm::dyn_cast<CompoundTerm>(term)};
  if (!applied) {
    SayExpected(state_, *term, "a name applied to arguments");
    return false;
  }
  // Later arguments see the variables numbered so far.
  NumberVariables(*term, numbers_);
  arguments.append(applied->arguments().begin(), applied->arguments().end());
  return true;
}

const OmpDirective *DirectiveConverter::Convert() {
  if (KindAt(0) != TokenKind::Name) {
    if (KindAt(0) == TokenKind::Eof) {
      outer_.messages.Say(outer_.tokens.tokens[directive_].spelling,
          "expected an OpenMP directive after '%s'"_err_en_US, "%$omp");
    } else {
      outer_.messages.Say(At(0).spelling,
          "expected an OpenMP directive, but found '%s'"_err_en_US,
          At(0).spelling.str());
    }
    return nullptr;
  }
  // The longest run of names that is a directive.
  std::uint32_t names{0};
  while (KindAt(names) == TokenKind::Name) {
    ++names;
  }
  llvm::omp::Directive id{llvm::omp::Directive::OMPD_unknown};
  std::uint32_t at{names};
  std::string spelling;
  for (; at > 0; --at) {
    spelling.clear();
    for (std::uint32_t j{0}; j < at; ++j) {
      spelling += (j > 0 ? " " : "") + UnquoteSpelling(At(j).spelling);
    }
    id = llvm::omp::getOpenMPDirectiveKind(spelling);
    if (id != llvm::omp::Directive::OMPD_unknown) {
      break;
    }
  }
  if (id == llvm::omp::Directive::OMPD_unknown) {
    outer_.messages.Say(At(0).spelling,
        "'%s' is not an OpenMP directive"_err_en_US,
        UnquoteSpelling(At(0).spelling));
    return nullptr;
  }
  CharBlock name{At(0).spelling.begin(), At(at - 1).spelling.end()};
  if (!llvm::to_underlying(llvm::omp::getDirectiveLanguages(id) &
          llvm::omp::SourceLanguage::Fortran)) {
    // Mercury follows Fortran: parallel do, not parallel for.
    std::string fortran{spelling};
    if (llvm::StringRef{fortran}.ends_with("for")) {
      fortran.replace(fortran.size() - 3, 3, "do");
    }
    if (fortran != spelling &&
        llvm::omp::getOpenMPDirectiveKind(fortran) !=
            llvm::omp::Directive::OMPD_unknown) {
      outer_.messages.Say(name,
          "'%s' is C's form of the OpenMP directive; Mercury uses Fortran's, '%s'"_err_en_US,
          spelling, fortran);
    } else {
      outer_.messages.Say(name,
          "'%s' is an OpenMP directive of C only; Mercury uses Fortran's directives"_err_en_US,
          spelling);
    }
    return nullptr;
  }
  // The directive's arguments, as in critical(Name).
  llvm::SmallVector<const Term *, 2> argumentTerms;
  --at;
  if (!ReadApplied(at, argumentTerms)) {
    return nullptr;
  }
  auto arguments{Elements(dataTerm, argumentTerms)};
  llvm::SmallVector<OmpClause, 4> clauses;
  bool ok{arguments.has_value()};
  bool afterClause{false};
  while (KindAt(at) != TokenKind::Eof) {
    if (afterClause && KindAt(at) == TokenKind::Comma) {
      ++at;
    }
    if (std::optional<OmpClause> clause{Clause(at)}) {
      clauses.push_back(*clause);
      afterClause = true;
    } else {
      // The tokens of a clause that failed are not known.
      return nullptr;
    }
  }
  if (!ok) {
    return nullptr;
  }
  return state_.arena.Make<OmpDirective>(TokenRange{0, at - 1}, id,
      *arguments,
      state_.arena.SaveArray(llvm::ArrayRef<OmpClause>{clauses}));
}

std::optional<OmpClause> DirectiveConverter::Clause(std::uint32_t &at) {
  std::uint32_t first{at};
  const Token &token{At(at)};
  if (token.kind == TokenKind::Eof) {
    state_.messages.Say(token.spelling,
        "expected an OpenMP clause, but found the end of the directive"_err_en_US);
    return std::nullopt;
  }
  if (token.kind != TokenKind::Name) {
    state_.messages.Say(token.spelling,
        "expected an OpenMP clause, but found '%s'"_err_en_US,
        token.spelling.str());
    return std::nullopt;
  }
  std::string name{UnquoteSpelling(token.spelling)};
  llvm::SmallVector<const Term *, 4> argumentTerms;
  if (!ReadApplied(at, argumentTerms)) {
    return std::nullopt;
  }
  TokenRange tokens{first, at - 1};
  CharBlock source{token.spelling.begin(), At(at - 1).spelling.end()};
  auto make{[&](auto *node) -> std::optional<OmpClause> { return node; }};
  auto count{[&](std::size_t minimum, std::size_t maximum,
                 llvm::StringRef usage) {
    if (argumentTerms.size() < minimum || argumentTerms.size() > maximum) {
      state_.messages.Say(source, "expected %s"_err_en_US, usage.str());
      return false;
    }
    return true;
  }};
  if (name == "index") {
    if (!count(1, 1, "index(Variable)")) {
      return std::nullopt;
    }
    std::optional<const Variable *> variable{
        Apply(variableNode, *argumentTerms[0], state_)};
    if (!variable) {
      return std::nullopt;
    }
    return make(state_.arena.Make<OmpIndexClause>(tokens, *variable));
  }
  if (name == "upper") {
    if (!count(1, 1, "upper(Bound)")) {
      return std::nullopt;
    }
    std::optional<DataTerm> bound{Apply(dataTerm, *argumentTerms[0], state_)};
    if (!bound) {
      return std::nullopt;
    }
    return make(state_.arena.Make<OmpUpperClause>(tokens, *bound));
  }
  llvm::omp::Clause id{llvm::omp::getOpenMPClauseKind(name)};
  if (id == llvm::omp::Clause::OMPC_unknown) {
    state_.messages.Say(token.spelling,
        "'%s' is not an OpenMP clause"_err_en_US, name);
    return std::nullopt;
  }
  Arguments split{SplitArguments(argumentTerms)};
  switch (id) {
  case llvm::omp::Clause::OMPC_schedule: {
    if (split.rest.empty() || split.rest.size() > 2) {
      state_.messages.Say(source,
          "expected schedule([Modifiers :] Kind [, Chunk])"_err_en_US);
      return std::nullopt;
    }
    auto modifiers{Elements(scheduleModifier, split.modifiers)};
    std::optional<OmpScheduleKind> kind{
        Apply(scheduleKind, *split.rest[0], state_)};
    std::optional<DataTerm> chunk;
    bool ok{modifiers && kind};
    if (split.rest.size() == 2) {
      chunk = Apply(dataTerm, *split.rest[1], state_);
      ok &= chunk.has_value();
    }
    if (!ok) {
      return std::nullopt;
    }
    return make(state_.arena.Make<OmpScheduleClause>(
        tokens, *modifiers, *kind, chunk));
  }
  case llvm::omp::Clause::OMPC_reduction: {
    // Without a ':', the arguments are [Modifier,] Op.
    llvm::SmallVector<const Term *, 4> before{
        split.colon ? split.modifiers : split.rest};
    if (before.empty() || before.size() > 2) {
      state_.messages.Say(source,
          "expected reduction([Modifier,] Op [: Variables])"_err_en_US);
      return std::nullopt;
    }
    std::optional<OmpReductionModifier> modifier;
    bool ok{true};
    if (before.size() == 2) {
      modifier = Apply(reductionModifier, *before[0], state_);
      ok &= modifier.has_value();
    }
    std::optional<const SymName *> op{Apply(symName, *before.back(), state_)};
    std::optional<llvm::ArrayRef<const Variable *>> variables{
        llvm::ArrayRef<const Variable *>{}};
    if (split.colon) {
      variables = Elements(variableNode, split.rest);
    }
    if (!ok || !op || !variables) {
      return std::nullopt;
    }
    return make(state_.arena.Make<OmpReductionClause>(
        tokens, modifier, *op, *variables));
  }
  case llvm::omp::Clause::OMPC_map: {
    // map(always, to : X): the last modifier is the map type, if it is
    // one.
    llvm::ArrayRef<const Term *> modifierTerms{split.modifiers};
    std::optional<OmpMapType> type;
    bool ok{true};
    if (!modifierTerms.empty() && mapType.Matches(*modifierTerms.back())) {
      type = mapType.Convert(*modifierTerms.back(), state_);
      modifierTerms = modifierTerms.drop_back();
    }
    auto modifiers{Elements(mapModifier, modifierTerms)};
    auto variables{Elements(variableNode, split.rest)};
    if (split.rest.empty()) {
      state_.messages.Say(
          source, "expected map([Modifiers,] [Type :] Variables)"_err_en_US);
      ok = false;
    }
    if (!ok || !modifiers || !variables) {
      return std::nullopt;
    }
    return make(state_.arena.Make<OmpMapClause>(
        tokens, *modifiers, type, *variables));
  }
  default: {
    auto modifiers{Elements(dataTerm, split.modifiers)};
    auto arguments{Elements(dataTerm, split.rest)};
    if (!modifiers || !arguments) {
      return std::nullopt;
    }
    return make(state_.arena.Make<OmpGenericClause>(
        tokens, id, *modifiers, *arguments));
  }
  }
}

} // namespace

std::optional<Goal> ConvertOmpGoal(
    const DirectiveTerm &t, ConvertState &state) {
  const OmpDirective *directive{
      DirectiveConverter{t.directiveToken(), state}.Convert()};
  std::optional<Goal> annotated{Apply(goal, *t.annotated(), state)};
  if (!directive || !annotated) {
    return std::nullopt;
  }
  return state.arena.Make<OmpGoal>(t.tokens(), directive, *annotated);
}

} // namespace merkurius::parser
