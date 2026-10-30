//===-- lib/Parser/convert-goals.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The converter's rules for clauses, goals, DCG goals and data-terms. The
// parse tree keeps each form as it is written: DCG rules, state variables
// and record syntax are expanded later, by the transformations the
// reference manual defines over these forms.
//
// Nearly every goal and special data-term is recognized by its top functor
// and arity, so each category is one function that tests for them in turn;
// a term that is none of them is a call or a functor term.

#include "convert-rules.h"

namespace merkurius::parser {

namespace {

struct DcgGoalRule {
  using resultType = DcgGoal;
  bool Matches(const Term &) const { return true; }
  std::optional<DcgGoal> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a DCG goal"; }
};
// Term, or Term::Mode.
struct ModedTermRule {
  using resultType = const ModedTerm *;
  bool Matches(const Term &) const { return true; }
  std::optional<resultType> Convert(const Term &, ConvertState &) const;
  void Expected(llvm::raw_ostream &os) const { os << "a data-term"; }
};

constexpr DcgGoalRule dcgGoal;
constexpr ModedTermRule modedTerm;

constexpr auto fieldSpec{
    expect("a field name", construct<FieldSpec>(appliedName(dataTerm)))};
constexpr auto call{construct<Call>(appliedName(dataTerm))};
constexpr auto dcgCall{construct<DcgCall>(appliedName(dataTerm))};

std::optional<Purity> PurityNamed(llvm::StringRef name) {
  if (name == "impure") {
    return Purity::Impure;
  } else if (name == "semipure") {
    return Purity::Semipure;
  }
  return std::nullopt;
}

// The single argument of a compound term.
const Term &Operand(const CompoundTerm &c, std::size_t j = 0) {
  return *c.arguments()[j];
}

std::optional<DataTerm> ConvertStateVar(
    const CompoundTerm &c, StateVarKind kind, ConvertState &state) {
  const Term &name{StripParentheses(Operand(c))};
  if (const auto *var{llvm::dyn_cast<VariableTerm>(&name)}) {
    return state.arena.Make<StateVar>(
        c.tokens(), kind, state.arena.Save(var->name()));
  }
  SayExpected(state, name, "a state variable's name");
  return std::nullopt;
}

// A variable, or a state variable !X, !.X or !:X.
struct QuantifiedVarRule {
  using resultType = DataTerm;
  bool Matches(const Term &t) const {
    return llvm::isa<VariableTerm>(t) || AsCompound(t, "!", 1) ||
        AsCompound(t, "!.", 1) || AsCompound(t, "!:", 1);
  }
  std::optional<DataTerm> Convert(const Term &t, ConvertState &state) const {
    return dataTerm.Convert(t, state);
  }
  void Expected(llvm::raw_ostream &os) const {
    os << "a variable or a state variable";
  }
};

// !X, the variable of io(!X) and state(M, !X) in a trace goal.
struct StateVarRule {
  using resultType = DataTerm;
  bool Matches(const Term &t) const { return AsCompound(t, "!", 1); }
  std::optional<DataTerm> Convert(const Term &t, ConvertState &state) const {
    return dataTerm.Convert(t, state);
  }
  void Expected(llvm::raw_ostream &os) const { os << "a state variable !X"; }
};

// X, a plain variable.
struct PlainVarRule {
  using resultType = DataTerm;
  bool Matches(const Term &t) const { return llvm::isa<VariableTerm>(t); }
  std::optional<DataTerm> Convert(const Term &t, ConvertState &state) const {
    return dataTerm.Convert(t, state);
  }
  void Expected(llvm::raw_ostream &os) const { os << "a variable"; }
};

constexpr auto quantifiedGoalVars{expect(
    "a list of variables", listOf(QuantifiedVarRule{}))};

// The fields of Term ^ Field1 ^ Field2, which nest on the right.
std::optional<llvm::ArrayRef<const FieldSpec *>> ConvertFields(
    const Term &term, ConvertState &state) {
  llvm::SmallVector<const Term *, 4> fields;
  auto collect{[&](const Term &t, auto &self) -> void {
    if (const CompoundTerm * caret{AsCompound(t, "^", 2)}) {
      self(Operand(*caret, 0), self);
      self(Operand(*caret, 1), self);
    } else {
      fields.push_back(&t);
    }
  }};
  collect(term, collect);
  return detail::ConvertElements(
      fieldSpec, llvm::ArrayRef<const Term *>{fields}, state);
}

// The operands of the terms with functor `name`/2, looking through
// parentheses: a, b, c for (a, b), c.
void Flatten(const Term &term, llvm::StringRef name,
    llvm::SmallVectorImpl<const Term *> &into) {
  const Term &t{StripParentheses(term)};
  if (const CompoundTerm * c{AsCompound(t, name, 2)}) {
    Flatten(Operand(*c, 0), name, into);
    Flatten(Operand(*c, 1), name, into);
  } else {
    into.push_back(&t);
  }
}

// The disjuncts of G1 ; G2 ; G3, where a disjunct C -> T ; E is an
// if-then-else instead.
bool IsIfThenElseArrow(const Term &t) {
  const CompoundTerm *semicolon{AsCompound(t, ";", 2)};
  return semicolon && AsCompound(Operand(*semicolon), "->", 2);
}

void FlattenDisjunction(
    const Term &term, llvm::SmallVectorImpl<const Term *> &into) {
  const Term &t{StripParentheses(term)};
  const CompoundTerm *semicolon{AsCompound(t, ";", 2)};
  if (semicolon && !IsIfThenElseArrow(t)) {
    FlattenDisjunction(Operand(*semicolon, 0), into);
    FlattenDisjunction(Operand(*semicolon, 1), into);
  } else {
    into.push_back(&t);
  }
}

// The parts of if C then T else E, or C -> T ; E.
struct IfThenElseParts {
  IfThenElseForm form;
  const Term *condition, *then, *otherwise;
};

std::optional<IfThenElseParts> MatchIfThenElse(const Term &t) {
  if (const CompoundTerm * elseTerm{AsCompound(t, "else", 2)}) {
    if (const CompoundTerm * ifTerm{AsCompound(Operand(*elseTerm), "if", 1)}) {
      if (const CompoundTerm * thenTerm{
              AsCompound(Operand(*ifTerm), "then", 2)}) {
        return IfThenElseParts{IfThenElseForm::IfThenElse,
            &Operand(*thenTerm, 0), &Operand(*thenTerm, 1),
            &Operand(*elseTerm, 1)};
      }
    }
  }
  if (IsIfThenElseArrow(t)) {
    const auto &semicolon{llvm::cast<CompoundTerm>(t)};
    const CompoundTerm &arrow{*AsCompound(Operand(semicolon), "->", 2)};
    return IfThenElseParts{IfThenElseForm::Arrow, &Operand(arrow, 0),
        &Operand(arrow, 1), &Operand(semicolon, 1)};
  }
  return std::nullopt;
}

// An if-then-else without its else part, which is an error.
bool IsIfThenWithoutElse(const Term &t) {
  const CompoundTerm *ifTerm{AsCompound(t, "if", 1)};
  return AsCompound(t, "->", 2) ||
      (ifTerm && AsCompound(Operand(*ifTerm), "then", 2));
}

void SayMissingElse(const Term &t, ConvertState &state) {
  state.messages.Say(GetSource(t, state.tokens),
      "an if-then-else must have an else part"_err_en_US);
}

// The parts of a lambda expression (reference manual, "Lambda
// expressions"): Purity pred(Args) is Det :- Body, with --> for a DCG body,
// or Purity func(Args) = Result is Det :- Body, where the purity, the
// determinism and the body are optional; a predicate needs a determinism.
struct LambdaParts {
  Purity purity{Purity::Pure};
  PredOrFunc kind{PredOrFunc::Predicate};
  llvm::ArrayRef<const Term *> arguments;
  const Term *result{nullptr};
  const Term *determinism{nullptr};
  const Term *body{nullptr};
  bool dcg{false};
  // pred(...) or func(...) with a body or a determinism, but not in one of
  // the forms above.
  bool malformed{false};
};

std::optional<LambdaParts> MatchLambda(const Term &term) {
  LambdaParts parts;
  const Term *t{&StripParentheses(term)};
  auto peelPurity{[&]() {
    if (const auto *c{llvm::dyn_cast<CompoundTerm>(t)}; c && Arity(*c) == 1) {
      if (std::optional<Purity> purity{PurityNamed(FunctorName(*c))}) {
        parts.purity = *purity;
        t = &StripParentheses(Operand(*c));
      }
    }
  }};
  peelPurity();
  if (const CompoundTerm * rule{AsCompound(*t, ":-", 2)}) {
    parts.body = &Operand(*rule, 1);
    t = &StripParentheses(Operand(*rule, 0));
  } else if (const CompoundTerm * dcg{AsCompound(*t, "-->", 2)}) {
    parts.body = &Operand(*dcg, 1);
    parts.dcg = true;
    t = &StripParentheses(Operand(*dcg, 0));
  }
  peelPurity();
  bool layered{parts.body != nullptr};
  if (const CompoundTerm * is{AsCompound(*t, "is", 2)}) {
    layered = true;
    parts.determinism = &Operand(*is, 1);
    t = &StripParentheses(Operand(*is, 0));
  }
  if (const CompoundTerm * equals{AsCompound(*t, "=", 2)}) {
    parts.result = &Operand(*equals, 1);
    t = &StripParentheses(Operand(*equals, 0));
  }
  llvm::StringRef name{FunctorName(*t)};
  if (name == "func") {
    parts.kind = PredOrFunc::Function;
  } else if (name != "pred") {
    return std::nullopt;
  }
  if (const auto *compound{llvm::dyn_cast<CompoundTerm>(t)}) {
    parts.arguments = compound->arguments();
  }
  bool isFunction{parts.kind == PredOrFunc::Function};
  if (isFunction ? !parts.result || parts.dcg
                 : parts.result || !parts.determinism) {
    if (!layered) {
      return std::nullopt; // an ordinary functor named pred or func
    }
    parts.malformed = true;
  }
  return parts;
}

std::optional<DataTerm> ConvertLambda(
    const Term &t, const LambdaParts &parts, ConvertState &state) {
  if (parts.malformed) {
    state.messages.Say(GetSource(t, state.tokens),
        "a lambda expression's head must be 'pred(Args) is Det' or "
        "'func(Args) = Result', with an optional 'is Det'"_err_en_US);
    return std::nullopt;
  }
  auto arguments{detail::ConvertElements(modedTerm, parts.arguments, state)};
  bool ok{arguments.has_value()};
  const ModedTerm *result{nullptr};
  ok &= ConvertOptional(modedTerm, parts.result, result, state);
  std::optional<Determinism> det;
  ok &= ConvertOptional(determinism, parts.determinism, det, state);
  std::optional<Goal> body;
  std::optional<DcgGoal> dcgBody;
  if (parts.dcg) {
    ok &= ConvertOptional(dcgGoal, parts.body, dcgBody, state);
  } else {
    ok &= ConvertOptional(goal, parts.body, body, state);
  }
  if (!ok) {
    return std::nullopt;
  }
  return state.arena.Make<Lambda>(t.tokens(), parts.purity, parts.kind,
      *arguments, result, det, body, dcgBody);
}

} // namespace

//===----------------------------------------------------------------------===//
// Data-terms
//===----------------------------------------------------------------------===//

std::optional<DataTerm> DataTermRule::Convert(
    const Term &t, ConvertState &state) const {
  auto make{[&](auto *node) -> std::optional<DataTerm> { return node; }};
  auto elements{[&](llvm::ArrayRef<const Term *> terms) {
    return detail::ConvertElements(dataTerm, terms, state);
  }};
  switch (t.kind()) {
  case Term::Kind::Variable: {
    const auto &var{llvm::cast<VariableTerm>(t)};
    return make(state.arena.Make<Variable>(
        t.tokens(), state.arena.Save(var.name()), var.number()));
  }
  case Term::Kind::Integer:
  case Term::Kind::Float:
  case Term::Kind::String:
    if (auto value{literal.Convert(t, state)}) {
      return make(*value);
    }
    return std::nullopt;
  case Term::Kind::ImplementationDefinedLiteral:
    return make(state.arena.Make<ImplementationLiteral>(t.tokens(),
        state.arena.Save(
            llvm::cast<ImplementationDefinedLiteralTerm>(t).name())));
  case Term::Kind::List: {
    const auto &list{llvm::cast<ListTerm>(t)};
    auto items{elements(list.elements())};
    std::optional<DataTerm> tail;
    bool ok{ConvertOptional(dataTerm, list.tail(), tail, state)};
    if (!items || !ok) {
      return std::nullopt;
    }
    return make(state.arena.Make<ListData>(t.tokens(), *items, tail));
  }
  case Term::Kind::Tuple:
    if (auto items{elements(llvm::cast<TupleTerm>(t).elements())}) {
      return make(state.arena.Make<TupleData>(t.tokens(), *items));
    }
    return std::nullopt;
  case Term::Kind::Apply: {
    const auto &apply{llvm::cast<ApplyTerm>(t)};
    std::optional<DataTerm> function{
        Apply(dataTerm, *apply.closure(), state)};
    auto arguments{elements(apply.arguments())};
    if (!function || !arguments) {
      return std::nullopt;
    }
    return make(
        state.arena.Make<ApplyData>(t.tokens(), *function, *arguments));
  }
  case Term::Kind::Backquote: {
    // X `with_type` T is a type qualification, X `f` Y is f(X, Y), and
    // X `F` Y, with a variable F, is F(X, Y).
    const auto &backquote{llvm::cast<BackquoteTerm>(t)};
    const Term &op{StripParentheses(*backquote.op())};
    const Term *operands[]{backquote.left(), backquote.right()};
    if (FunctorName(op) == "with_type") {
      std::optional<DataTerm> term{Apply(dataTerm, *operands[0], state)};
      std::optional<Type> ty{Apply(type, *operands[1], state)};
      if (!term || !ty) {
        return std::nullopt;
      }
      return make(state.arena.Make<TypeQualified>(t.tokens(), *term, *ty));
    }
    auto arguments{elements(operands)};
    if (llvm::isa<VariableTerm>(op)) {
      std::optional<DataTerm> function{dataTerm.Convert(op, state)};
      if (!arguments || !function) {
        return std::nullopt;
      }
      return make(
          state.arena.Make<ApplyData>(t.tokens(), *function, *arguments));
    }
    std::optional<const SymName *> name{symName.Convert(op, state)};
    if (!arguments || !name) {
      return std::nullopt;
    }
    return make(state.arena.Make<FunctorData>(t.tokens(), *name, *arguments));
  }
  case Term::Kind::Directive:
    return Apply(
        dataTerm, IgnoreDirective(llvm::cast<DirectiveTerm>(t), state), state);
  case Term::Kind::Parenthesized: // Apply looks through parentheses
  case Term::Kind::Name:
  case Term::Kind::Compound:
    break;
  }
  // The special data-terms (reference manual, "Data-terms").
  if (const CompoundTerm * c{AsCompound(t, "!", 1)}) {
    return ConvertStateVar(*c, StateVarKind::Both, state);
  } else if (const CompoundTerm * c{AsCompound(t, "!.", 1)}) {
    return ConvertStateVar(*c, StateVarKind::Current, state);
  } else if (const CompoundTerm * c{AsCompound(t, "!:", 1)}) {
    return ConvertStateVar(*c, StateVarKind::Next, state);
  }
  if (std::optional<IfThenElseParts> ite{MatchIfThenElse(t)}) {
    std::optional<Goal> condition{Apply(goal, *ite->condition, state)};
    std::optional<DataTerm> then{Apply(dataTerm, *ite->then, state)};
    std::optional<DataTerm> otherwise{Apply(dataTerm, *ite->otherwise, state)};
    if (!condition || !then || !otherwise) {
      return std::nullopt;
    }
    return make(state.arena.Make<ConditionalData>(
        t.tokens(), ite->form, *condition, *then, *otherwise));
  }
  if (IsIfThenWithoutElse(t)) {
    SayMissingElse(t, state);
    return std::nullopt;
  }
  if (std::optional<LambdaParts> lambda{MatchLambda(t)}) {
    return ConvertLambda(t, *lambda, state);
  }
  if (const CompoundTerm * at{AsCompound(t, "@", 2)}) {
    std::optional<DataTerm> left{Apply(dataTerm, Operand(*at, 0), state)};
    std::optional<DataTerm> right{Apply(dataTerm, Operand(*at, 1), state)};
    if (!left || !right) {
      return std::nullopt;
    }
    return make(state.arena.Make<UnifyData>(t.tokens(), *left, *right));
  }
  if (const CompoundTerm * assign{AsCompound(t, ":=", 2)}) {
    if (const CompoundTerm * caret{AsCompound(Operand(*assign), "^", 2)}) {
      std::optional<DataTerm> term{Apply(dataTerm, Operand(*caret, 0), state)};
      auto fields{ConvertFields(Operand(*caret, 1), state)};
      std::optional<DataTerm> value{
          Apply(dataTerm, Operand(*assign, 1), state)};
      if (!term || !fields || !value) {
        return std::nullopt;
      }
      return make(state.arena.Make<FieldUpdate>(
          t.tokens(), *term, *fields, *value));
    }
  }
  if (const CompoundTerm * caret{AsCompound(t, "^", 2)}) {
    std::optional<DataTerm> term{Apply(dataTerm, Operand(*caret, 0), state)};
    auto fields{ConvertFields(Operand(*caret, 1), state)};
    if (!term || !fields) {
      return std::nullopt;
    }
    return make(state.arena.Make<FieldSelection>(t.tokens(), *term, *fields));
  }
  if (auto operands{BinaryOperands(t, ":")}) {
    std::optional<DataTerm> term{Apply(dataTerm, *operands->first, state)};
    std::optional<Type> ty{Apply(type, *operands->second, state)};
    if (!term || !ty) {
      return std::nullopt;
    }
    return make(state.arena.Make<TypeQualified>(t.tokens(), *term, *ty));
  }
  if (const CompoundTerm * c{AsCompound(t, "coerce", 1)}) {
    if (std::optional<DataTerm> term{Apply(dataTerm, Operand(*c), state)}) {
      return make(state.arena.Make<Coerce>(t.tokens(), *term));
    }
    return std::nullopt;
  }
  // A functor term: a name, or a compound term such as f(X) or X + Y.
  if (auto converted{appliedName(dataTerm).Convert(t, state)}) {
    auto [name, arguments]{*converted};
    return make(state.arena.Make<FunctorData>(t.tokens(), name, arguments));
  }
  return std::nullopt;
}

std::optional<const ModedTerm *> ModedTermRule::Convert(
    const Term &t, ConvertState &state) const {
  if (const CompoundTerm * colons{AsCompound(t, "::", 2)}) {
    std::optional<DataTerm> term{Apply(dataTerm, Operand(*colons, 0), state)};
    std::optional<Mode> m{Apply(mode, Operand(*colons, 1), state)};
    if (!term || !m) {
      return std::nullopt;
    }
    return state.arena.Make<ModedTerm>(t.tokens(), *term, m);
  }
  if (std::optional<DataTerm> term{Apply(dataTerm, t, state)}) {
    return state.arena.Make<ModedTerm>(t.tokens(), *term);
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Goals
//===----------------------------------------------------------------------===//

namespace {

// A trace goal's compile_time or run_time condition: and, or and not over
// primitive conditions (reference manual, "Trace goals"). At compile time
// these are flag("Name"), tracelevel(shallow) or tracelevel(deep), and
// grade(Test); at run time, env("Name").
struct TraceConditionRule {
  using resultType = const TraceCondition *;
  bool runTime;
  bool Matches(const Term &) const { return true; }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    auto make{[&](TraceConditionKind kind,
                  llvm::ArrayRef<const TraceCondition *> operands,
                  std::optional<llvm::StringRef> value)
        -> std::optional<resultType> {
      return state.arena.Make<TraceCondition>(
          t.tokens(), kind, operands, value);
    }};
    std::optional<TraceConditionKind> kind;
    if (AsCompound(t, "and", 2)) {
      kind = TraceConditionKind::And;
    } else if (AsCompound(t, "or", 2)) {
      kind = TraceConditionKind::Or;
    } else if (AsCompound(t, "not", 1)) {
      kind = TraceConditionKind::Not;
    }
    if (kind) {
      if (auto converted{detail::ConvertElements(
              *this, llvm::cast<CompoundTerm>(t).arguments(), state)}) {
        return make(*kind, *converted, std::nullopt);
      }
      return std::nullopt;
    }
    const auto *c{llvm::dyn_cast<CompoundTerm>(&t)};
    const Term *argument{c && Arity(*c) == 1 ? &Operand(*c) : nullptr};
    llvm::StringRef name{c ? FunctorName(*c) : ""};
    if (runTime ? name != "env"
                : !llvm::is_contained({"flag", "tracelevel", "grade"}, name)) {
      SayExpected(state, t, detail::ExpectedText(*this));
      return std::nullopt;
    }
    if (!argument) {
      SayExpected(state, t, detail::ExpectedText(*this));
      return std::nullopt;
    }
    const Term &a{StripParentheses(*argument)};
    if (name == "flag" || name == "env") {
      // mmc: "compile_time parameter `flag' takes a string as argument".
      if (auto text{Apply(expect("a string", stringLiteral), a, state)}) {
        return make(name == "flag" ? TraceConditionKind::Flag
                                   : TraceConditionKind::Env,
            {}, *text);
      }
      return std::nullopt;
    }
    llvm::StringRef value{llvm::isa<NameTerm>(a) ? FunctorName(a) : ""};
    if (name == "tracelevel") {
      if (value != "shallow" && value != "deep") {
        SayExpected(state, a, "'shallow' or 'deep'");
        return std::nullopt;
      }
      return make(TraceConditionKind::TraceLevel, {}, value);
    }
    // The manual's grade tests, and rbmm, which mmc also accepts.
    if (!llvm::is_contained({"debug", "ssdebug", "prof", "profdeep", "par",
                                "trail", "rbmm", "llds", "mlds", "c",
                                "csharp", "java"},
            value)) {
      SayExpected(state, a,
          "a grade test: 'debug', 'ssdebug', 'prof', 'profdeep', 'par', "
          "'trail', 'rbmm', 'llds', 'mlds', 'c', 'csharp' or 'java'");
      return std::nullopt;
    }
    return make(TraceConditionKind::Grade, {}, value);
  }
  void Expected(llvm::raw_ostream &os) const {
    os << (runTime ? "a run_time condition: env(\"Name\"), with and, or "
                     "and not"
                   : "a compile_time condition: flag(\"Name\"), "
                     "tracelevel(Level) or grade(Test), with and, or and "
                     "not");
  }
};

struct TraceParamRule {
  using resultType = const TraceParam *;
  bool Matches(const Term &) const { return true; }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    const CompoundTerm *condition{AsCompound(t, "compile_time", 1)};
    TraceParamKind kind{TraceParamKind::CompileTime};
    if (!condition && (condition = AsCompound(t, "run_time", 1))) {
      kind = TraceParamKind::RunTime;
    }
    if (condition) {
      if (auto converted{Apply(
              TraceConditionRule{kind == TraceParamKind::RunTime},
              Operand(*condition), state)}) {
        return state.arena.Make<TraceParam>(
            t.tokens(), kind, *converted, nullptr);
      }
      return std::nullopt;
    }
    if (const CompoundTerm * io{AsCompound(t, "io", 1)}) {
      if (auto var{Apply(StateVarRule{}, Operand(*io), state)}) {
        return state.arena.Make<TraceParam>(
            t.tokens(), TraceParamKind::Io, nullptr, nullptr, *var);
      }
      return std::nullopt;
    }
    if (const CompoundTerm * c{AsCompound(t, "state", 2)}) {
      auto name{Apply(
          expect("a mutable's name", symName), Operand(*c, 0), state)};
      auto var{Apply(StateVarRule{}, Operand(*c, 1), state)};
      if (!name || !var) {
        return std::nullopt;
      }
      return state.arena.Make<TraceParam>(
          t.tokens(), TraceParamKind::State, nullptr, *name, *var);
    }
    SayExpected(state, t, detail::ExpectedText(*this));
    return std::nullopt;
  }
  void Expected(llvm::raw_ostream &os) const {
    os << "a trace parameter: compile_time(...), run_time(...), io(!IO) or "
          "state(Mutable, !S)";
  }
};

struct AtomicParamRule {
  using resultType = const AtomicParam *;
  bool Matches(const Term &) const { return true; }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    for (auto [label, kind] :
        {std::pair{"outer", AtomicParamKind::Outer},
            std::pair{"inner", AtomicParamKind::Inner}}) {
      if (const CompoundTerm * c{AsCompound(t, label, 1)}) {
        if (auto var{Apply(StateVarRule{}, Operand(*c), state)}) {
          return state.arena.Make<AtomicParam>(t.tokens(), kind, *var);
        }
        return std::nullopt;
      }
      if (const CompoundTerm * c{AsCompound(t, label, 2)}) {
        auto initial{Apply(PlainVarRule{}, Operand(*c, 0), state)};
        auto final{Apply(PlainVarRule{}, Operand(*c, 1), state)};
        if (!initial || !final) {
          return std::nullopt;
        }
        return state.arena.Make<AtomicParam>(
            t.tokens(), kind, *initial, *final);
      }
    }
    if (const CompoundTerm * c{AsCompound(t, "vars", 1)}) {
      if (auto vars{Apply(quantifiedVars, Operand(*c), state)}) {
        return state.arena.Make<AtomicParam>(t.tokens(),
            AtomicParamKind::Vars, std::nullopt, std::nullopt, *vars);
      }
      return std::nullopt;
    }
    SayExpected(state, t, detail::ExpectedText(*this));
    return std::nullopt;
  }
  void Expected(llvm::raw_ostream &os) const {
    os << "an atomic parameter: outer(!S), inner(!S), outer(S0, S), "
          "inner(S0, S) or vars([Vars])";
  }
};

// io(!S), the one kind of try parameter.
struct TryParamRule {
  using resultType = DataTerm;
  bool Matches(const Term &) const { return true; }
  std::optional<DataTerm> Convert(const Term &t, ConvertState &state) const {
    if (const CompoundTerm * io{AsCompound(t, "io", 1)}) {
      return Apply(StateVarRule{}, Operand(*io), state);
    }
    SayExpected(state, t, "a try parameter io(!S)");
    return std::nullopt;
  }
  void Expected(llvm::raw_ostream &os) const {
    os << "a try parameter io(!S)";
  }
};

// The parts of try Params Goal then T else E catch P1 -> C1 catch P2 -> C2
// catch_any V -> A, which nest outward from the try: catch_any(catch(else(
// then(try(Params, Goal), T), E), catch(P1 -> C1, P2 -> C2)), V -> A).
struct TryParts {
  const Term *params, *goal;
  const Term *then{nullptr}, *otherwise{nullptr};
  llvm::SmallVector<const Term *, 4> catches; // P -> C, in order
  const Term *catchAny{nullptr};
};

std::optional<TryParts> MatchTry(const Term &term) {
  TryParts parts;
  const Term *t{&StripParentheses(term)};
  if (const CompoundTerm * c{AsCompound(*t, "catch_any", 2)}) {
    parts.catchAny = &Operand(*c, 1);
    t = &StripParentheses(Operand(*c, 0));
  }
  // catch is right-associative: G catch (P1 -> C1 catch P2 -> C2).
  if (const CompoundTerm * c{AsCompound(*t, "catch", 2)}) {
    Flatten(Operand(*c, 1), "catch", parts.catches);
    t = &StripParentheses(Operand(*c, 0));
  }
  if (const CompoundTerm * c{AsCompound(*t, "else", 2)}) {
    const CompoundTerm *then{AsCompound(Operand(*c, 0), "then", 2)};
    if ((then && AsCompound(Operand(*then, 0), "try", 2)) ||
        AsCompound(Operand(*c, 0), "try", 2)) {
      parts.otherwise = &Operand(*c, 1);
      t = &StripParentheses(Operand(*c, 0));
    }
  }
  if (const CompoundTerm * c{AsCompound(*t, "then", 2)};
      c && AsCompound(Operand(*c, 0), "try", 2)) {
    parts.then = &Operand(*c, 1);
    t = &StripParentheses(Operand(*c, 0));
  }
  // try(Params, Goal) alone is a call to try/2, as in mmc.
  const CompoundTerm *tryTerm{AsCompound(*t, "try", 2)};
  if (!tryTerm ||
      (!parts.then && !parts.otherwise && parts.catches.empty() &&
          !parts.catchAny)) {
    return std::nullopt;
  }
  parts.params = &Operand(*tryTerm, 0);
  parts.goal = &Operand(*tryTerm, 1);
  return parts;
}

// Pattern -> Goal, after catch or catch_any.
std::optional<const CatchClause *> ConvertCatch(
    const Term &t, ConvertState &state) {
  const CompoundTerm *arrow{AsCompound(t, "->", 2)};
  if (!arrow) {
    SayExpected(state, StripParentheses(t), "'Pattern -> Goal'");
    return std::nullopt;
  }
  std::optional<DataTerm> pattern{Apply(dataTerm, Operand(*arrow, 0), state)};
  std::optional<Goal> handler{Apply(goal, Operand(*arrow, 1), state)};
  if (!pattern || !handler) {
    return std::nullopt;
  }
  return state.arena.Make<CatchClause>(
      StripParentheses(t).tokens(), *pattern, *handler);
}

std::optional<Goal> ConvertTry(
    const Term &t, const TryParts &parts, ConvertState &state) {
  if (!parts.then) {
    // mmc requires the then part.
    state.messages.Say(GetSource(t, state.tokens),
        "a try goal must have a then part: 'try Params Goal then Goal', "
        "then optionally 'else Goal', 'catch Pattern -> Goal' and "
        "'catch_any Var -> Goal'"_err_en_US);
    return std::nullopt;
  }
  bool ok{true};
  llvm::ArrayRef<DataTerm> io;
  ok &= ConvertOptional(
      expect("a list of try parameters", listOf(TryParamRule{})),
      parts.params, io, state);
  std::optional<Goal> body{Apply(goal, *parts.goal, state)};
  ok &= body.has_value();
  std::optional<Goal> then{Apply(goal, *parts.then, state)};
  ok &= then.has_value();
  std::optional<Goal> otherwise;
  ok &= ConvertOptional(goal, parts.otherwise, otherwise, state);
  llvm::SmallVector<const CatchClause *, 4> catches;
  for (const Term *c : parts.catches) {
    if (auto converted{ConvertCatch(*c, state)}) {
      catches.push_back(*converted);
    } else {
      ok = false;
    }
  }
  const CatchClause *catchAny{nullptr};
  if (parts.catchAny) {
    auto converted{ConvertCatch(*parts.catchAny, state)};
    ok &= converted.has_value();
    catchAny = converted.value_or(nullptr);
  }
  if (!ok) {
    return std::nullopt;
  }
  return state.arena.Make<TryGoal>(t.tokens(), io, *body, *then, otherwise,
      state.arena.SaveArray(llvm::ArrayRef<const CatchClause *>{catches}),
      catchAny);
}

// [X], the variable of a switch requirement.
std::optional<DataTerm> ConvertSwitchVar(const Term &t, ConvertState &state) {
  auto vars{Apply(expect("a list of one variable",
                      listOf(expect("a variable", QuantifiedVarRule{}))),
      t, state)};
  if (!vars) {
    return std::nullopt;
  }
  if (vars->size() != 1) {
    SayExpected(state, StripParentheses(t), "a list of one variable");
    return std::nullopt;
  }
  return vars->front();
}

} // namespace

std::optional<Goal> GoalRule::Convert(
    const Term &t, ConvertState &state) const {
  auto make{[&](auto *node) -> std::optional<Goal> { return node; }};
  auto both{[&](const CompoundTerm &c, auto &rule) {
    auto left{Apply(rule, Operand(c, 0), state)};
    auto right{Apply(rule, Operand(c, 1), state)};
    return std::pair{left, right};
  }};
  if (const auto *directive{llvm::dyn_cast<DirectiveTerm>(&t)}) {
    return ConvertOmpGoal(*directive, state);
  }
  // Higher-order calls: P, and P(X, Y).
  if (llvm::isa<VariableTerm>(t)) {
    if (auto closure{dataTerm.Convert(t, state)}) {
      return make(state.arena.Make<HigherOrderCall>(
          t.tokens(), *closure, llvm::ArrayRef<DataTerm>{}));
    }
    return std::nullopt;
  }
  if (const auto *apply{llvm::dyn_cast<ApplyTerm>(&t)}) {
    auto closure{Apply(dataTerm, *apply->closure(), state)};
    auto arguments{
        detail::ConvertElements(dataTerm, apply->arguments(), state)};
    if (!closure || !arguments) {
      return std::nullopt;
    }
    return make(
        state.arena.Make<HigherOrderCall>(t.tokens(), *closure, *arguments));
  }
  // X `p` Y is p(X, Y), and X `P` Y, with a variable P, is P(X, Y).
  if (const auto *backquote{llvm::dyn_cast<BackquoteTerm>(&t)}) {
    const Term &op{StripParentheses(*backquote->op())};
    const Term *operands[]{backquote->left(), backquote->right()};
    auto arguments{detail::ConvertElements(
        dataTerm, llvm::ArrayRef<const Term *>{operands}, state)};
    if (llvm::isa<VariableTerm>(op)) {
      auto closure{dataTerm.Convert(op, state)};
      if (!closure || !arguments) {
        return std::nullopt;
      }
      return make(state.arena.Make<HigherOrderCall>(
          t.tokens(), *closure, *arguments));
    }
    auto name{symName.Convert(op, state)};
    if (!name || !arguments) {
      return std::nullopt;
    }
    return make(state.arena.Make<Call>(t.tokens(), *name, *arguments));
  }
  // {G1, G2} is a call to '{}'/2, as in mmc.
  if (const auto *tuple{llvm::dyn_cast<TupleTerm>(&t)}) {
    auto arguments{detail::ConvertElements(dataTerm, tuple->elements(), state)};
    if (!arguments) {
      return std::nullopt;
    }
    const SymName *braces{MakeBracketSymName(t, "{}", state)};
    return make(state.arena.Make<Call>(t.tokens(), braces, *arguments));
  }
  if (!llvm::isa<NameTerm>(t) && !llvm::isa<CompoundTerm>(t)) {
    SayExpected(state, t, "a goal");
    return std::nullopt;
  }
  llvm::StringRef name{FunctorName(t)};
  std::size_t arity{Arity(t)};
  // !S ^ Fields := Value, which updates a state variable.
  if (const CompoundTerm * update{AsCompound(t, ":=", 2)}) {
    const CompoundTerm *caret{AsCompound(Operand(*update, 0), "^", 2)};
    if (caret && AsCompound(StripParentheses(Operand(*caret, 0)), "!", 1)) {
      auto var{dataTerm.Convert(StripParentheses(Operand(*caret, 0)), state)};
      auto fields{ConvertFields(Operand(*caret, 1), state)};
      auto value{Apply(dataTerm, Operand(*update, 1), state)};
      if (!var || !fields || !value) {
        return std::nullopt;
      }
      return make(state.arena.Make<StateVarFieldUpdate>(
          t.tokens(), *var, *fields, *value));
    }
  }
  // The DCG goals =(T), :=(T), T =^ F and ^ F := T, which mmc rejects
  // outside DCG rules.
  if ((arity == 1 && (name == "=" || name == ":=")) ||
      (arity == 2 && (name == "=^" || name == ":="))) {
    state.messages.Say(GetSource(t, state.tokens),
        "'%s' is a DCG goal, which may appear only in a DCG rule"_err_en_US,
        Excerpt(t, state.tokens));
    return std::nullopt;
  }
  if (arity == 0 && (name == "true" || name == "fail")) {
    return make(state.arena.Make<BuiltinGoal>(t.tokens(),
        name == "true" ? BuiltinGoalKind::True : BuiltinGoalKind::Fail));
  }
  const auto *c{llvm::dyn_cast<CompoundTerm>(&t)};
  if (c && arity == 2 && (name == "," || name == "&")) {
    llvm::SmallVector<const Term *, 8> conjuncts;
    Flatten(t, name, conjuncts);
    if (auto goals{detail::ConvertElements(
            goal, llvm::ArrayRef<const Term *>{conjuncts}, state)}) {
      return make(
          state.arena.Make<Conjunction>(t.tokens(), name == "&", *goals));
    }
    return std::nullopt;
  }
  if (std::optional<IfThenElseParts> ite{MatchIfThenElse(t)}) {
    std::optional<Goal> condition{Apply(goal, *ite->condition, state)};
    std::optional<Goal> then{Apply(goal, *ite->then, state)};
    std::optional<Goal> otherwise{Apply(goal, *ite->otherwise, state)};
    if (!condition || !then || !otherwise) {
      return std::nullopt;
    }
    return make(state.arena.Make<IfThenElse>(
        t.tokens(), ite->form, *condition, *then, *otherwise));
  }
  if (c && arity == 2 && name == ";") {
    llvm::SmallVector<const Term *, 8> disjuncts;
    FlattenDisjunction(t, disjuncts);
    if (auto goals{detail::ConvertElements(
            goal, llvm::ArrayRef<const Term *>{disjuncts}, state)}) {
      return make(state.arena.Make<Disjunction>(t.tokens(), *goals));
    }
    return std::nullopt;
  }
  if (IsIfThenWithoutElse(t)) {
    SayMissingElse(t, state);
    return std::nullopt;
  }
  if (c && arity == 1 && (name == "not" || name == "\\+")) {
    if (auto negated{Apply(goal, Operand(*c), state)}) {
      return make(state.arena.Make<Negation>(t.tokens(), *negated));
    }
    return std::nullopt;
  }
  if (c && arity == 2) {
    std::optional<ImplicationKind> implication{
        llvm::StringSwitch<std::optional<ImplicationKind>>(name)
            .Case("=>", ImplicationKind::Implies)
            .Case("<=", ImplicationKind::ImpliedBy)
            .Case("<=>", ImplicationKind::Equivalent)
            .Default(std::nullopt)};
    if (implication) {
      auto [left, right]{both(*c, goal)};
      if (!left || !right) {
        return std::nullopt;
      }
      return make(state.arena.Make<Implication>(
          t.tokens(), *implication, *left, *right));
    }
    if (name == "=" || name == "\\=") {
      auto [left, right]{both(*c, dataTerm)};
      if (!left || !right) {
        return std::nullopt;
      }
      if (name == "=") {
        return make(state.arena.Make<Unification>(t.tokens(), *left, *right));
      }
      return make(state.arena.Make<Inequality>(t.tokens(), *left, *right));
    }
    if (name == "some" || name == "all") {
      auto vars{Apply(quantifiedGoalVars, Operand(*c, 0), state)};
      auto quantified{Apply(goal, Operand(*c, 1), state)};
      if (!vars || !quantified) {
        return std::nullopt;
      }
      return make(state.arena.Make<QuantifiedGoal>(t.tokens(),
          name == "some" ? Quantifier::Some : Quantifier::All, *vars,
          *quantified));
    }
    std::optional<PromiseSolutionsKind> promise{
        llvm::StringSwitch<std::optional<PromiseSolutionsKind>>(name)
            .Case("promise_equivalent_solutions",
                PromiseSolutionsKind::EquivalentSolutions)
            .Case("promise_equivalent_solution_sets",
                PromiseSolutionsKind::EquivalentSolutionSets)
            .Case("arbitrary", PromiseSolutionsKind::Arbitrary)
            .Default(std::nullopt)};
    if (promise) {
      auto vars{Apply(quantifiedGoalVars, Operand(*c, 0), state)};
      auto promised{Apply(goal, Operand(*c, 1), state)};
      if (!vars || !promised) {
        return std::nullopt;
      }
      return make(state.arena.Make<PromiseSolutions>(
          t.tokens(), *promise, *vars, *promised));
    }
    llvm::StringRef arms{name};
    if (name == "require_complete_switch" ||
        (arms.consume_front("require_switch_arms_") &&
            DeterminismRule::FromName(arms))) {
      std::optional<Determinism> armDeterminism;
      if (name != "require_complete_switch") {
        armDeterminism = DeterminismRule::FromName(arms);
      }
      auto var{ConvertSwitchVar(Operand(*c, 0), state)};
      auto required{Apply(goal, Operand(*c, 1), state)};
      if (!var || !required) {
        return std::nullopt;
      }
      return make(state.arena.Make<RequireSwitch>(
          t.tokens(), armDeterminism, *var, *required));
    }
    if (name == "disable_warnings" || name == "disable_warning") {
      auto warnings{Apply(expect("a list of warning names",
                              listOf(expect("a warning name", symName))),
          Operand(*c, 0), state)};
      auto quiet{Apply(goal, Operand(*c, 1), state)};
      if (!warnings || !quiet) {
        return std::nullopt;
      }
      llvm::SmallVector<llvm::StringRef, 4> names;
      for (const SymName *warning : *warnings) {
        names.push_back(warning->name);
      }
      return make(state.arena.Make<DisableWarnings>(t.tokens(),
          state.arena.SaveArray(llvm::ArrayRef<llvm::StringRef>{names}),
          *quiet));
    }
    if (name == "trace") {
      auto params{Apply(expect("a list of trace parameters",
                            listOf(TraceParamRule{})),
          Operand(*c, 0), state)};
      auto traced{Apply(goal, Operand(*c, 1), state)};
      if (!params || !traced) {
        return std::nullopt;
      }
      // mmc: "duplicate compile_time trace parameter"; state may repeat,
      // once for each mutable.
      for (std::size_t j{0}; j < params->size(); ++j) {
        TraceParamKind kind{(*params)[j]->kind};
        if (kind != TraceParamKind::State &&
            llvm::any_of(params->take_front(j),
                [&](const TraceParam *p) { return p->kind == kind; })) {
          state.messages.Say(GetSource(Operand(*c, 0), state.tokens),
              "the trace parameter '%s' is given more than once"_err_en_US,
              kind == TraceParamKind::CompileTime ? "compile_time"
                  : kind == TraceParamKind::RunTime ? "run_time"
                                                    : "io");
          return std::nullopt;
        }
      }
      return make(state.arena.Make<TraceGoal>(t.tokens(), *params, *traced));
    }
    if (name == "atomic") {
      auto params{Apply(expect("a list of atomic parameters",
                            listOf(AtomicParamRule{})),
          Operand(*c, 0), state)};
      // G1 or_else G2 ..., which nest either way.
      llvm::SmallVector<Goal, 2> goals;
      bool ok{params.has_value()};
      auto alternatives{[&](const Term &a, auto &self) -> void {
        const Term &bare{StripParentheses(a)};
        if (const CompoundTerm * orElse{AsCompound(bare, "or_else", 2)}) {
          self(Operand(*orElse, 0), self);
          self(Operand(*orElse, 1), self);
        } else if (auto alternative{Apply(goal, a, state)}) {
          goals.push_back(*alternative);
        } else {
          ok = false;
        }
      }};
      alternatives(Operand(*c, 1), alternatives);
      if (!ok) {
        return std::nullopt;
      }
      // mmc: "duplicate outer atomic parameter", and "Atomic goal is
      // missing a specification of the inner STM state".
      static constexpr const char *labels[]{"outer", "inner", "vars"};
      bool given[3]{};
      for (const AtomicParam *param : *params) {
        auto j{static_cast<std::size_t>(param->kind)};
        if (given[j]) {
          state.messages.Say(GetSource(Operand(*c, 0), state.tokens),
              "the atomic parameter '%s' is given more than once"_err_en_US,
              labels[j]);
          return std::nullopt;
        }
        given[j] = true;
      }
      if (!given[0] || !given[1]) {
        state.messages.Say(GetSource(Operand(*c, 0), state.tokens),
            "an atomic goal needs the parameters outer(...) and inner(...), "
            "but %s missing"_err_en_US,
            !given[0] && !given[1] ? "both are"
                : !given[0]        ? "outer(...) is"
                                   : "inner(...) is");
        return std::nullopt;
      }
      return make(state.arena.Make<AtomicGoal>(t.tokens(), *params,
          state.arena.SaveArray(llvm::ArrayRef<Goal>{goals})));
    }
  }
  if (std::optional<TryParts> tryParts{MatchTry(t)}) {
    return ConvertTry(t, *tryParts, state);
  }
  if (c && arity == 1) {
    if (std::optional<Purity> purity{PurityNamed(name)}) {
      if (auto impure{Apply(goal, Operand(*c), state)}) {
        return make(
            state.arena.Make<PurityGoal>(t.tokens(), *purity, *impure));
      }
      return std::nullopt;
    }
    std::optional<Purity> promised{
        llvm::StringSwitch<std::optional<Purity>>(name)
            .Case("promise_pure", Purity::Pure)
            .Case("promise_semipure", Purity::Semipure)
            .Case("promise_impure", Purity::Impure)
            .Default(std::nullopt)};
    if (promised) {
      if (auto promising{Apply(goal, Operand(*c), state)}) {
        return make(state.arena.Make<PromisePurity>(
            t.tokens(), *promised, *promising));
      }
      return std::nullopt;
    }
    llvm::StringRef required{name};
    if (required.consume_front("require_")) {
      if (std::optional<Determinism> det{
              DeterminismRule::FromName(required)}) {
        if (auto checked{Apply(goal, Operand(*c), state)}) {
          return make(state.arena.Make<RequireDeterminism>(
              t.tokens(), *det, *checked));
        }
        return std::nullopt;
      }
    }
    if (name == "event") {
      const Term &event{StripParentheses(Operand(*c))};
      if (call.Matches(event)) {
        if (auto converted{call.Convert(event, state)}) {
          return make(state.arena.Make<EventGoal>(t.tokens(), *converted));
        }
        return std::nullopt;
      }
      SayExpected(state, event, "a predicate call");
      return std::nullopt;
    }
  }
  // Any other goal is a call (reference manual, "Goals").
  if (auto converted{call.Convert(t, state)}) {
    return make(*converted);
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// DCG goals
//===----------------------------------------------------------------------===//

std::optional<DcgGoal> DcgGoalRule::Convert(
    const Term &t, ConvertState &state) const {
  auto make{[&](auto *node) -> std::optional<DcgGoal> { return node; }};
  if (const auto *directive{llvm::dyn_cast<DirectiveTerm>(&t)}) {
    return Apply(dcgGoal, IgnoreDirective(*directive, state), state);
  }
  if (llvm::isa<VariableTerm>(t)) {
    if (auto closure{dataTerm.Convert(t, state)}) {
      return make(state.arena.Make<DcgHigherOrderCall>(
          t.tokens(), *closure, llvm::ArrayRef<DataTerm>{}));
    }
    return std::nullopt;
  }
  if (const auto *apply{llvm::dyn_cast<ApplyTerm>(&t)}) {
    auto closure{Apply(dataTerm, *apply->closure(), state)};
    auto arguments{
        detail::ConvertElements(dataTerm, apply->arguments(), state)};
    if (!closure || !arguments) {
      return std::nullopt;
    }
    return make(state.arena.Make<DcgHigherOrderCall>(
        t.tokens(), *closure, *arguments));
  }
  // { Goal }: the reader reads { G1, G2 } as a tuple of two, which here is
  // their conjunction.
  if (const auto *tuple{llvm::dyn_cast<TupleTerm>(&t)};
      tuple && !tuple->elements().empty()) {
    auto goals{detail::ConvertElements(goal, tuple->elements(), state)};
    if (!goals) {
      return std::nullopt;
    }
    Goal braced{goals->size() == 1
            ? goals->front()
            : Goal{state.arena.Make<Conjunction>(t.tokens(), false, *goals)}};
    return make(state.arena.Make<DcgBraces>(t.tokens(), braced));
  }
  if (const auto *list{llvm::dyn_cast<ListTerm>(&t)}) {
    if (list->tail()) {
      SayExpected(state, t, "a list without a tail");
      return std::nullopt;
    }
    if (auto terms{
            detail::ConvertElements(dataTerm, list->elements(), state)}) {
      return make(state.arena.Make<DcgTerminals>(t.tokens(), *terms));
    }
    return std::nullopt;
  }
  if (!llvm::isa<NameTerm>(t) && !llvm::isa<CompoundTerm>(t)) {
    SayExpected(state, t, "a DCG goal");
    return std::nullopt;
  }
  llvm::StringRef name{FunctorName(t)};
  std::size_t arity{Arity(t)};
  const auto *c{llvm::dyn_cast<CompoundTerm>(&t)};
  if (c && arity == 2 && (name == "," || name == "&")) {
    llvm::SmallVector<const Term *, 8> conjuncts;
    Flatten(t, name, conjuncts);
    if (auto goals{detail::ConvertElements(
            dcgGoal, llvm::ArrayRef<const Term *>{conjuncts}, state)}) {
      return make(
          state.arena.Make<DcgConjunction>(t.tokens(), name == "&", *goals));
    }
    return std::nullopt;
  }
  if (std::optional<IfThenElseParts> ite{MatchIfThenElse(t)}) {
    auto condition{Apply(dcgGoal, *ite->condition, state)};
    auto then{Apply(dcgGoal, *ite->then, state)};
    auto otherwise{Apply(dcgGoal, *ite->otherwise, state)};
    if (!condition || !then || !otherwise) {
      return std::nullopt;
    }
    return make(state.arena.Make<DcgIfThenElse>(
        t.tokens(), ite->form, *condition, *then, *otherwise));
  }
  if (c && arity == 2 && name == ";") {
    llvm::SmallVector<const Term *, 8> disjuncts;
    FlattenDisjunction(t, disjuncts);
    if (auto goals{detail::ConvertElements(
            dcgGoal, llvm::ArrayRef<const Term *>{disjuncts}, state)}) {
      return make(state.arena.Make<DcgDisjunction>(t.tokens(), *goals));
    }
    return std::nullopt;
  }
  if (IsIfThenWithoutElse(t)) {
    SayMissingElse(t, state);
    return std::nullopt;
  }
  if (c && arity == 1 && (name == "not" || name == "\\+")) {
    if (auto negated{Apply(dcgGoal, Operand(*c), state)}) {
      return make(state.arena.Make<DcgNegation>(t.tokens(), *negated));
    }
    return std::nullopt;
  }
  if (c && arity == 2 && (name == "some" || name == "all")) {
    auto vars{Apply(quantifiedGoalVars, Operand(*c, 0), state)};
    auto quantified{Apply(dcgGoal, Operand(*c, 1), state)};
    if (!vars || !quantified) {
      return std::nullopt;
    }
    return make(state.arena.Make<DcgQuantified>(t.tokens(),
        name == "some" ? Quantifier::Some : Quantifier::All, *vars,
        *quantified));
  }
  if (c && arity == 1 && (name == "=" || name == ":=")) {
    if (auto term{Apply(dataTerm, Operand(*c), state)}) {
      return make(state.arena.Make<DcgUnify>(t.tokens(), name == ":=", *term));
    }
    return std::nullopt;
  }
  if (c && arity == 2 && name == "=^") {
    auto term{Apply(dataTerm, Operand(*c, 0), state)};
    auto fields{ConvertFields(Operand(*c, 1), state)};
    if (!term || !fields) {
      return std::nullopt;
    }
    return make(
        state.arena.Make<DcgFieldSelection>(t.tokens(), *term, *fields));
  }
  if (c && arity == 2 && name == ":=") {
    if (const CompoundTerm * caret{AsCompound(Operand(*c, 0), "^", 1)}) {
      auto fields{ConvertFields(Operand(*caret), state)};
      auto value{Apply(dataTerm, Operand(*c, 1), state)};
      if (!fields || !value) {
        return std::nullopt;
      }
      return make(
          state.arena.Make<DcgFieldUpdate>(t.tokens(), *fields, *value));
    }
  }
  // X = Y and X := Y would be calls to =/4 and :=/4; mmc rejects them.
  if (c && arity == 2 && (name == "=" || name == ":=")) {
    SayExpected(state, t,
        name == "=" ? "a DCG goal such as '=(Term)' or 'Term =^ Field'"
                    : "a DCG goal such as ':=(Term)' or '^ Field := Term'");
    return std::nullopt;
  }
  // Any other DCG goal is a call that gets the DCG arguments appended.
  if (auto converted{dcgCall.Convert(t, state)}) {
    return make(*converted);
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Clauses
//===----------------------------------------------------------------------===//

std::optional<Clause> ClauseRule::Convert(
    const Term &term, ConvertState &state) const {
  const Term &t{StripParentheses(term)};
  const Term *head{&t};
  const Term *body{nullptr};
  bool dcg{false};
  if (const CompoundTerm * rule{AsCompound(t, ":-", 2)}) {
    head = &Operand(*rule, 0);
    body = &Operand(*rule, 1);
  } else if (const CompoundTerm * rule{AsCompound(t, "-->", 2)}) {
    head = &Operand(*rule, 0);
    body = &Operand(*rule, 1);
    dcg = true;
  }
  const Term *result{nullptr};
  if (!dcg) {
    if (const CompoundTerm * equals{AsCompound(*head, "=", 2)}) {
      head = &Operand(*equals, 0);
      result = &Operand(*equals, 1);
    }
  }
  // The head is a name or a compound term, or, as in mmc, X `p` Y or a
  // tuple, which is a call to '{}'.
  const Term *headTerm{&StripParentheses(*head)};
  if (const auto *directive{llvm::dyn_cast<DirectiveTerm>(headTerm)}) {
    headTerm = &StripParentheses(IgnoreDirective(*directive, state));
  }
  const Term &h{*headTerm};
  std::optional<std::tuple<const SymName *, llvm::ArrayRef<const ModedTerm *>>>
      converted;
  auto withArguments{[&](std::optional<const SymName *> name,
                         llvm::ArrayRef<const Term *> terms) {
    auto arguments{detail::ConvertElements(modedTerm, terms, state)};
    if (name && arguments) {
      converted.emplace(*name, *arguments);
    }
  }};
  const auto *backquote{llvm::dyn_cast<BackquoteTerm>(&h)};
  if (backquote && !llvm::isa<VariableTerm>(StripParentheses(*backquote->op()))) {
    const Term *operands[]{backquote->left(), backquote->right()};
    withArguments(Apply(symName, *backquote->op(), state),
        llvm::ArrayRef<const Term *>{operands});
  } else if (const auto *tuple{llvm::dyn_cast<TupleTerm>(&h)}) {
    withArguments(MakeBracketSymName(h, "{}", state), tuple->elements());
  } else if (llvm::isa<NameTerm>(h) || llvm::isa<CompoundTerm>(h)) {
    converted = result ? functionHead(modedTerm).Convert(h, state)
                       : appliedName(modedTerm).Convert(h, state);
  } else {
    SayExpected(state, h, "a clause head");
    return std::nullopt;
  }
  bool ok{converted.has_value()};
  if (dcg) {
    std::optional<DcgGoal> dcgBody{Apply(dcgGoal, *body, state)};
    if (!ok || !dcgBody) {
      return std::nullopt;
    }
    auto [name, arguments]{*converted};
    return state.arena.Make<DcgRule>(t.tokens(), name, arguments, *dcgBody);
  }
  const ModedTerm *resultTerm{nullptr};
  ok &= ConvertOptional(modedTerm, result, resultTerm, state);
  if (ok) {
    // mmc: "some but not all arguments have mode annotations".
    auto arguments{std::get<1>(*converted)};
    std::size_t withModes{0}, all{arguments.size()};
    for (const ModedTerm *argument : arguments) {
      withModes += argument->mode.has_value();
    }
    if (resultTerm) {
      withModes += resultTerm->mode.has_value();
      ++all;
    }
    if (withModes != 0 && withModes != all) {
      state.messages.Say(GetSource(h, state.tokens),
          "either every argument%s or none must have a mode"_err_en_US,
          resultTerm ? " and the result" : "");
      ok = false;
    }
  }
  std::optional<Goal> goalBody;
  ok &= ConvertOptional(goal, body, goalBody, state);
  if (!ok) {
    return std::nullopt;
  }
  auto [name, arguments]{*converted};
  return state.arena.Make<FactOrRule>(t.tokens(),
      result ? PredOrFunc::Function : PredOrFunc::Predicate, name, arguments,
      resultTerm, goalBody);
}

} // namespace merkurius::parser
