//===-- lib/Parser/convert-pragmas.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The converter's rules for pragmas (reference manual, "Pragmas", "Foreign
// language interface" and "Implementation-dependent extensions"). A pragma
// is ':- pragma Name(Arguments)', where Name decides the arguments' forms;
// the checks here are the ones mmc makes while reading a pragma.

#include "convert-rules.h"

namespace merkurius::parser {

namespace {

// An enumerator's name in snake case: TabledForIo is tabled_for_io.
std::string SnakeName(std::string_view camel) {
  std::string snake;
  for (char ch : camel) {
    if (ch >= 'A' && ch <= 'Z') {
      if (!snake.empty()) {
        snake += '_';
      }
      snake += static_cast<char>(ch - 'A' + 'a');
    } else {
      snake += ch;
    }
  }
  return snake;
}

// The enumerator of E, which has N of them, whose snake case name is `name`.
template <typename E, std::size_t N>
std::optional<E> FromSnakeName(llvm::StringRef name) {
  for (std::size_t j{0}; j < N; ++j) {
    E e{static_cast<E>(j)};
    if (SnakeName(EnumToString(e)) == name) {
      return e;
    }
  }
  return std::nullopt;
}

// Any term, unconverted.
struct AnyTermRule {
  using resultType = const Term *;
  bool Matches(const Term &) const { return true; }
  std::optional<const Term *> Convert(const Term &t, ConvertState &) const {
    return &t;
  }
  void Expected(llvm::raw_ostream &os) const { os << "a term"; }
};

// A name that is one of E's enumerators in snake case.
template <typename E, std::size_t N> class EnumNameRule {
public:
  using resultType = E;
  constexpr explicit EnumNameRule(llvm::StringLiteral expected)
      : expected_{expected} {}
  bool Matches(const Term &t) const {
    return llvm::isa<NameTerm>(t) && FromSnakeName<E, N>(FunctorName(t));
  }
  std::optional<E> Convert(const Term &t, ConvertState &) const {
    return FromSnakeName<E, N>(FunctorName(t));
  }
  void Expected(llvm::raw_ostream &os) const { os << expected_; }

private:
  llvm::StringLiteral expected_;
};

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

constexpr StringRule stringLiteral;

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

constexpr LanguageRule language;

// p/1, pred(p/1) or func(f/1).
struct PredNameArityRule {
  using resultType = const PredNameArity *;
  bool Matches(const Term &t) const {
    return nameAndArity.Matches(t) || AsCompound(t, "pred", 1) ||
        AsCompound(t, "func", 1);
  }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    std::optional<PredOrFunc> kind;
    const Term *nameArity{&t};
    if (!nameAndArity.Matches(t)) {
      kind = FunctorName(t) == "func" ? PredOrFunc::Function
                                      : PredOrFunc::Predicate;
      nameArity = llvm::cast<CompoundTerm>(t).arguments()[0];
    }
    auto converted{Apply(nameAndArity, *nameArity, state)};
    if (!converted) {
      return std::nullopt;
    }
    return state.arena.Make<PredNameArity>(t.tokens(), kind, *converted);
  }
  void Expected(llvm::raw_ostream &os) const {
    os << "'Name/Arity', 'pred(Name/Arity)' or 'func(Name/Arity)'";
  }
};

constexpr PredNameArityRule predNameArity;

// p(in, out) or f(in) = out; p alone has no arguments.
struct ProcSpecRule {
  using resultType = const ProcSpec *;
  bool Matches(const Term &t) const {
    return (llvm::isa<NameTerm>(t) || llvm::isa<CompoundTerm>(t)) &&
        !predNameArity.Matches(t);
  }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    const Term *head{&t};
    const Term *result{nullptr};
    if (const CompoundTerm * equals{AsCompound(t, "=", 2)}) {
      head = &StripParentheses(*equals->arguments()[0]);
      result = equals->arguments()[1];
    }
    auto converted{Apply(appliedName(mode), *head, state)};
    std::optional<Mode> resultMode;
    bool ok{ConvertOptional(mode, result, resultMode, state)};
    if (!converted || !ok) {
      return std::nullopt;
    }
    auto [name, modes]{*converted};
    return state.arena.Make<ProcSpec>(t.tokens(),
        result ? PredOrFunc::Function : PredOrFunc::Predicate, name, modes,
        resultMode);
  }
  void Expected(llvm::raw_ostream &os) const {
    os << "a predicate or function mode such as 'p(in, out)'";
  }
};

constexpr ProcSpecRule procSpec;
constexpr auto predSpec{anyOf<PredSpec>(predNameArity, procSpec)};

constexpr auto replacements{
    expect("a list of replacements", listOf(nameAndArity))};

// T = Type.
struct TypeSubstitutionRule {
  using resultType = const TypeSubstitution *;
  bool Matches(const Term &t) const {
    const CompoundTerm *equals{AsCompound(t, "=", 2)};
    return equals &&
        llvm::isa<VariableTerm>(StripParentheses(*equals->arguments()[0]));
  }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    const auto &equals{llvm::cast<CompoundTerm>(t)};
    const auto &var{
        llvm::cast<VariableTerm>(StripParentheses(*equals.arguments()[0]))};
    std::optional<Type> ty{Apply(type, *equals.arguments()[1], state)};
    if (!ty) {
      return std::nullopt;
    }
    return state.arena.Make<TypeSubstitution>(
        t.tokens(), state.arena.Save(var.name()), *ty);
  }
  void Expected(llvm::raw_ostream &os) const {
    os << "a type substitution 'T = Type'";
  }
};

// include_file("Path"), or the code itself.
std::optional<const ForeignCode *> ConvertForeignCode(
    const Term &term, ConvertState &state) {
  const Term &t{StripParentheses(term)};
  const Term *text{&t};
  const CompoundTerm *include{AsCompound(t, "include_file", 1)};
  if (include) {
    text = include->arguments()[0];
  }
  auto converted{Apply(include ? expect("a file name", stringLiteral)
                               : expect("a string or 'include_file(Path)'",
                                     stringLiteral),
      *text, state)};
  if (!converted) {
    return std::nullopt;
  }
  return state.arena.Make<ForeignCode>(t.tokens(), include != nullptr,
      *converted);
}

// Var::Mode.
std::optional<const ForeignProcArg *> ConvertForeignProcArg(
    const Term &term, ConvertState &state) {
  const Term &t{StripParentheses(term)};
  const CompoundTerm *colons{AsCompound(t, "::", 2)};
  const Term *var{
      colons ? &StripParentheses(*colons->arguments()[0]) : nullptr};
  if (!var || !llvm::isa<VariableTerm>(*var)) {
    SayExpected(state, t, "'Var::Mode'");
    return std::nullopt;
  }
  std::optional<Mode> m{Apply(mode, *colons->arguments()[1], state)};
  if (!m) {
    return std::nullopt;
  }
  const auto &v{llvm::cast<VariableTerm>(*var)};
  return state.arena.Make<ForeignProcArg>(t.tokens(),
      state.arena.Make<Variable>(
          var->tokens(), state.arena.Save(v.name()), v.number()),
      *m);
}

// Constructor - "Value".
struct ForeignEnumValueRule {
  using resultType = const ForeignEnumValue *;
  bool Matches(const Term &t) const {
    const CompoundTerm *pair{AsCompound(t, "-", 2)};
    return pair &&
        llvm::isa<StringTerm>(StripParentheses(*pair->arguments()[1]));
  }
  std::optional<resultType> Convert(
      const Term &t, ConvertState &state) const {
    const auto &pair{llvm::cast<CompoundTerm>(t)};
    auto constructor{Apply(
        expect("a constructor name", symName), *pair.arguments()[0], state)};
    if (!constructor) {
      return std::nullopt;
    }
    return state.arena.Make<ForeignEnumValue>(t.tokens(), *constructor,
        state.arena.Save(
            llvm::cast<StringTerm>(StripParentheses(*pair.arguments()[1]))
                .value()));
  }
  void Expected(llvm::raw_ostream &os) const {
    os << "'Constructor - \"Value\"'";
  }
};

constexpr auto enumValues{listOf(ForeignEnumValueRule{})};
constexpr auto typeNameArity{expect("a type name/arity", nameAndArity)};

// The groups of foreign_proc attributes of which at most one may be given,
// each a run of the enumeration from `first` to `last`.
struct AttributeGroup {
  ForeignProcAttribute first, last;
  bool Contains(ForeignProcAttribute a) const {
    return a >= first && a <= last;
  }
};

constexpr AttributeGroup exclusiveAttributes[]{
    {ForeignProcAttribute::MayCallMercury,
        ForeignProcAttribute::WillNotCallMercury},
    {ForeignProcAttribute::PromisePure, ForeignProcAttribute::PromiseSemipure},
    {ForeignProcAttribute::ThreadSafe, ForeignProcAttribute::MaybeThreadSafe},
    {ForeignProcAttribute::Terminates,
        ForeignProcAttribute::DoesNotTerminate},
    {ForeignProcAttribute::MayModifyTrail,
        ForeignProcAttribute::WillNotModifyTrail},
    {ForeignProcAttribute::MayCallMmTabled,
        ForeignProcAttribute::WillNotCallMmTabled},
    {ForeignProcAttribute::AffectsLiveness,
        ForeignProcAttribute::DoesNotAffectLiveness},
    {ForeignProcAttribute::MayDuplicate,
        ForeignProcAttribute::MayNotDuplicate},
    {ForeignProcAttribute::MayExportBody,
        ForeignProcAttribute::MayNotExportBody},
};

// Reports two attributes of the same group, as mmc does.
bool CheckForeignProcAttributes(const Term &t,
    llvm::ArrayRef<ForeignProcAttribute> attributes, ConvertState &state) {
  for (const AttributeGroup &group : exclusiveAttributes) {
    std::optional<ForeignProcAttribute> seen;
    for (ForeignProcAttribute attribute : attributes) {
      // As in mmc, an attribute may be repeated.
      if (!group.Contains(attribute) || attribute == seen) {
        continue;
      }
      if (seen) {
        state.messages.Say(GetSource(t, state.tokens),
            "the attributes '%s' and '%s' conflict"_err_en_US,
            SnakeName(EnumToString(*seen)), SnakeName(EnumToString(attribute)));
        return false;
      }
      seen = attribute;
    }
  }
  return true;
}

constexpr auto foreignProcAttributes{expect("a list of attributes",
    listOf(EnumNameRule<ForeignProcAttribute, ForeignProcAttribute_enumSize>{
        "a foreign_proc attribute"}))};

// The attribute list of a tabling pragma.
std::optional<const TablingAttributes *> ConvertTablingAttributes(
    const Term &term, TablingMethod method, ConvertState &state) {
  const Term &t{StripParentheses(term)};
  const auto *list{llvm::dyn_cast<ListTerm>(&t)};
  if (!list || list->tail()) {
    SayExpected(state, t, "a list of tabling attributes");
    return std::nullopt;
  }
  TablingAttributes attributes{t.tokens(), false, false, false, false,
      std::nullopt, std::nullopt, std::nullopt};
  bool ok{true};
  constexpr auto tabledArgs{expect("a list of 'value', 'addr', "
                                   "'promise_implied' or 'output'",
      listOf(EnumNameRule<TabledArg, TabledArg_enumSize>{
          "'value', 'addr', 'promise_implied' or 'output'"}))};
  for (const Term *element : list->elements()) {
    const Term &a{StripParentheses(*element)};
    llvm::StringRef name{FunctorName(a)};
    bool duplicate{false};
    auto flag{[&](bool &into) {
      duplicate = into;
      into = true;
    }};
    if (llvm::isa<NameTerm>(a) && name == "allow_reset") {
      flag(attributes.allowReset);
    } else if (llvm::isa<NameTerm>(a) && name == "statistics") {
      flag(attributes.statistics);
    } else if (llvm::isa<NameTerm>(a) && name == "fast_loose") {
      flag(attributes.fastLoose);
    } else if (llvm::isa<NameTerm>(a) &&
        name == "disable_warning_if_ignored") {
      flag(attributes.disableWarningIfIgnored);
      // The manual: loop_check and minimal_model change the program's
      // meaning, so they must not be ignored silently.
      if (method != TablingMethod::Memo) {
        state.messages.Say(GetSource(a, state.tokens),
            "'disable_warning_if_ignored' is allowed only in a memo "
            "pragma"_err_en_US);
        ok = false;
      }
    } else if (const auto *c{llvm::dyn_cast<CompoundTerm>(&a)};
        c && name == "specified" && Arity(*c) <= 2) {
      duplicate = attributes.specified.has_value();
      auto args{Apply(tabledArgs, *c->arguments()[0], state)};
      ok &= args.has_value();
      attributes.specified = args.value_or(llvm::ArrayRef<TabledArg>{});
      if (Arity(*c) == 2) {
        const Term &hidden{StripParentheses(*c->arguments()[1])};
        llvm::StringRef hiddenName{
            llvm::isa<NameTerm>(hidden) ? FunctorName(hidden) : ""};
        if (hiddenName == "hidden_arg_value") {
          attributes.hiddenArg = HiddenArg::Value;
        } else if (hiddenName == "hidden_arg_addr") {
          attributes.hiddenArg = HiddenArg::Addr;
        } else {
          SayExpected(state, hidden, "'hidden_arg_value' or 'hidden_arg_addr'");
          ok = false;
        }
      }
    } else if (const auto *c{llvm::dyn_cast<CompoundTerm>(&a)};
        c && name == "size_limit" && Arity(*c) == 1) {
      duplicate = attributes.sizeLimit.has_value();
      const Term &limit{StripParentheses(*c->arguments()[0])};
      const auto *integer{llvm::dyn_cast<IntegerTerm>(&limit)};
      if (!integer || integer->negative() ||
          integer->size() != IntegerSize::Default ||
          integer->magnitude().getActiveBits() > 32) {
        SayExpected(state, limit, "a size limit");
        ok = false;
      } else {
        attributes.sizeLimit =
            static_cast<unsigned>(integer->magnitude().getZExtValue());
      }
    } else {
      SayExpected(state, a,
          "a tabling attribute: 'allow_reset', 'statistics', 'fast_loose', "
          "'specified(...)', 'size_limit(N)' or 'disable_warning_if_ignored'");
      ok = false;
      continue;
    }
    if (duplicate) {
      state.messages.Say(GetSource(a, state.tokens),
          "the attribute '%s' is given more than once"_err_en_US, name.str());
      ok = false;
    }
  }
  if (!ok) {
    return std::nullopt;
  }
  return state.arena.Make<TablingAttributes>(attributes);
}

// The attribute list of a foreign_export_enum: prefix("P") and uppercase.
bool ConvertExportEnumAttributes(const Term &term,
    std::optional<llvm::StringRef> &prefix, bool &uppercase,
    ConvertState &state) {
  const Term &t{StripParentheses(term)};
  const auto *list{llvm::dyn_cast<ListTerm>(&t)};
  if (!list || list->tail()) {
    SayExpected(state, t, "a list of attributes");
    return false;
  }
  bool ok{true};
  for (const Term *element : list->elements()) {
    const Term &a{StripParentheses(*element)};
    bool duplicate{false};
    if (const CompoundTerm * c{AsCompound(a, "prefix", 1)}) {
      duplicate = prefix.has_value();
      auto text{Apply(stringLiteral, *c->arguments()[0], state)};
      ok &= text.has_value();
      prefix = text.value_or(prefix.value_or(""));
    } else if (llvm::isa<NameTerm>(a) && FunctorName(a) == "uppercase") {
      duplicate = uppercase;
      uppercase = true;
    } else {
      SayExpected(state, a, "'prefix(\"Prefix\")' or 'uppercase'");
      ok = false;
      continue;
    }
    if (duplicate) {
      // mmc: "the prefix attribute may not occur more than once".
      state.messages.Say(GetSource(a, state.tokens),
          "the attribute '%s' is given more than once"_err_en_US,
          FunctorName(a).str());
      ok = false;
    }
  }
  return ok;
}

// The number of arguments a pragma takes, from `min` to `max`.
bool CheckArgumentCount(const Term &t, llvm::StringRef name,
    std::size_t count, std::size_t min, std::size_t max,
    ConvertState &state) {
  if (count >= min && count <= max) {
    return true;
  }
  static constexpr const char *numbers[]{
      "no", "one", "two", "three", "four", "five"};
  std::string expected{numbers[min]};
  if (max > min) {
    expected += (max == min + 1 ? " or " : " to ");
    expected += numbers[max];
  }
  expected += max == 1 ? " argument" : " arguments";
  state.messages.Say(GetSource(t, state.tokens),
      "a ':- pragma %s' declaration must have %s"_err_en_US, name.str(),
      expected);
  return false;
}

// The pragmas mmc accepts that the reference manual does not describe.
bool IsPragmaToDo(llvm::StringRef name) {
  return name == "require_tail_recursion" || name == "mode_check_clauses" ||
      name == "oisu";
}

} // namespace

std::optional<Item> PragmaRule::Convert(
    const Term &term, ConvertState &state) const {
  auto make{[&](auto *node) -> std::optional<Item> { return node; }};
  const Term &t{StripParentheses(term)};
  const Term *body{
      &StripParentheses(*llvm::cast<CompoundTerm>(t).arguments()[0])};
  // foreign_type(...) where Attributes.
  const Term *where{nullptr};
  if (const CompoundTerm * w{AsCompound(*body, "where", 2)}) {
    where = w->arguments()[1];
    body = &StripParentheses(*w->arguments()[0]);
  }
  if (!llvm::isa<NameTerm>(*body) && !llvm::isa<CompoundTerm>(*body)) {
    SayExpected(state, *body, "a pragma");
    return std::nullopt;
  }
  llvm::StringRef name{FunctorName(*body)};
  llvm::ArrayRef<const Term *> args;
  if (const auto *c{llvm::dyn_cast<CompoundTerm>(body)}) {
    args = c->arguments();
  }
  if (where && name != "foreign_type") {
    state.messages.Say(GetSource(t, state.tokens),
        "only a foreign_type pragma may have 'where' attributes"_err_en_US);
    return std::nullopt;
  }
  auto count{[&](std::size_t min, std::size_t max) {
    return CheckArgumentCount(*body, name, args.size(), min, max, state);
  }};
  auto arg{[&](std::size_t j) -> const Term & { return *args[j]; }};

  if (std::optional<PredPragmaKind> kind{
          FromSnakeName<PredPragmaKind, PredPragmaKind_enumSize>(name)}) {
    if (!count(1, 1)) {
      return std::nullopt;
    }
    if (auto pred{Apply(predNameArity, arg(0), state)}) {
      return make(state.arena.Make<PredPragma>(t.tokens(), *kind, *pred));
    }
    return std::nullopt;
  }
  if (name == "obsolete" || name == "obsolete_proc") {
    if (!count(1, 2)) {
      return std::nullopt;
    }
    std::optional<PredSpec> pred;
    if (name == "obsolete") {
      if (auto converted{Apply(predNameArity, arg(0), state)}) {
        pred = *converted;
      }
    } else if (auto converted{Apply(procSpec, arg(0), state)}) {
      pred = *converted;
    }
    llvm::ArrayRef<const NameAndArity *> replaced;
    bool ok{args.size() < 2 ||
        ConvertOptional(replacements, &arg(1), replaced, state)};
    if (!pred || !ok) {
      return std::nullopt;
    }
    return make(state.arena.Make<ObsoletePragma>(t.tokens(), *pred, replaced));
  }
  if (name == "type_spec") {
    if (!count(2, 2)) {
      return std::nullopt;
    }
    auto pred{Apply(predSpec, arg(0), state)};
    auto substitution{Apply(commaListOf(TypeSubstitutionRule{}), arg(1), state)};
    if (!pred || !substitution) {
      return std::nullopt;
    }
    return make(
        state.arena.Make<TypeSpecPragma>(t.tokens(), *pred, *substitution));
  }
  if (std::optional<TablingMethod> method{
          FromSnakeName<TablingMethod, TablingMethod_enumSize>(name)}) {
    if (!count(1, 2)) {
      return std::nullopt;
    }
    auto pred{Apply(predSpec, arg(0), state)};
    const TablingAttributes *attributes{nullptr};
    bool ok{true};
    if (args.size() == 2) {
      auto converted{ConvertTablingAttributes(arg(1), *method, state)};
      ok = converted.has_value();
      attributes = converted.value_or(nullptr);
    }
    if (!pred || !ok) {
      return std::nullopt;
    }
    return make(state.arena.Make<TablingPragma>(
        t.tokens(), *method, *pred, attributes));
  }
  if (name == "fact_table") {
    if (!count(2, 2)) {
      return std::nullopt;
    }
    auto pred{Apply(predNameArity, arg(0), state)};
    auto file{Apply(expect("a file name", stringLiteral), arg(1), state)};
    if (!pred || !file) {
      return std::nullopt;
    }
    return make(state.arena.Make<FactTablePragma>(t.tokens(), *pred, *file));
  }
  if (name == "require_feature_set") {
    if (!count(1, 1)) {
      return std::nullopt;
    }
    constexpr auto features{expect("a list of features",
        listOf(EnumNameRule<Feature, Feature_enumSize>{
            "a feature: 'concurrency', 'single_prec_float', "
            "'double_prec_float', 'memo', 'parallel_conj', 'trailing', "
            "'strict_sequential' or 'conservative_gc'"}))};
    if (auto converted{Apply(features, arg(0), state)}) {
      return make(
          state.arena.Make<RequireFeatureSetPragma>(t.tokens(), *converted));
    }
    return std::nullopt;
  }
  if (name == "source_file") {
    if (!count(1, 1)) {
      return std::nullopt;
    }
    if (auto file{Apply(expect("a file name string", stringLiteral), arg(0), state)}) {
      return make(state.arena.Make<SourceFilePragma>(t.tokens(), *file));
    }
    return std::nullopt;
  }

  // The foreign language interface, whose pragmas begin with the language.
  if (name == "foreign_decl") {
    if (!count(2, 3)) {
      return std::nullopt;
    }
    auto lang{Apply(language, arg(0), state)};
    bool local{false}, ok{true};
    if (args.size() == 3) {
      const Term &visibility{StripParentheses(arg(1))};
      llvm::StringRef v{
          llvm::isa<NameTerm>(visibility) ? FunctorName(visibility) : ""};
      local = v == "local";
      if (!local && v != "exported") {
        SayExpected(state, visibility, "'local' or 'exported'");
        ok = false;
      }
    }
    auto code{ConvertForeignCode(arg(args.size() - 1), state)};
    if (!lang || !ok || !code) {
      return std::nullopt;
    }
    return make(
        state.arena.Make<ForeignDeclPragma>(t.tokens(), *lang, local, *code));
  }
  if (name == "foreign_code") {
    if (!count(2, 2)) {
      return std::nullopt;
    }
    auto lang{Apply(language, arg(0), state)};
    auto code{ConvertForeignCode(arg(1), state)};
    if (!lang || !code) {
      return std::nullopt;
    }
    return make(state.arena.Make<ForeignCodePragma>(t.tokens(), *lang, *code));
  }
  if (name == "foreign_proc") {
    if (!count(4, 4)) {
      return std::nullopt;
    }
    auto lang{Apply(language, arg(0), state)};
    const Term *head{&StripParentheses(arg(1))};
    const Term *resultTerm{nullptr};
    if (const CompoundTerm * equals{AsCompound(*head, "=", 2)}) {
      head = &StripParentheses(*equals->arguments()[0]);
      resultTerm = equals->arguments()[1];
    }
    bool ok{true};
    const SymName *procName{nullptr};
    llvm::SmallVector<const ForeignProcArg *, 8> procArgs;
    if (!llvm::isa<NameTerm>(*head) && !llvm::isa<CompoundTerm>(*head)) {
      SayExpected(state, *head, "a predicate or function head");
      ok = false;
    } else {
      // The name, then each argument as Var::Mode.
      auto converted{Apply(appliedName(AnyTermRule{}), *head, state)};
      ok &= converted.has_value();
      if (converted) {
        procName = std::get<0>(*converted);
        for (const Term *a : std::get<1>(*converted)) {
          auto procArg{ConvertForeignProcArg(*a, state)};
          ok &= procArg.has_value();
          procArgs.push_back(procArg.value_or(nullptr));
        }
      }
    }
    const ForeignProcArg *result{nullptr};
    if (resultTerm) {
      auto converted{ConvertForeignProcArg(*resultTerm, state)};
      ok &= converted.has_value();
      result = converted.value_or(nullptr);
    }
    if (ok) {
      // The manual: "It is an error for a variable to occur more than once
      // in the argument list."
      llvm::SmallVector<llvm::StringRef, 8> names;
      for (const ForeignProcArg *a : procArgs) {
        names.push_back(a->var->name);
      }
      if (result) {
        names.push_back(result->var->name);
      }
      for (std::size_t j{0}; j < names.size(); ++j) {
        if (names[j] != "_" &&
            llvm::is_contained(llvm::ArrayRef(names).take_front(j), names[j])) {
          state.messages.Say(GetSource(arg(1), state.tokens),
              "the variable %s appears more than once"_err_en_US,
              names[j].str());
          ok = false;
          break;
        }
      }
    }
    auto attributes{Apply(foreignProcAttributes, arg(2), state)};
    ok &= attributes && CheckForeignProcAttributes(arg(2), *attributes, state);
    auto code{Apply(expect("the foreign code, a string", stringLiteral), arg(3),
        state)};
    if (!lang || !ok || !code) {
      return std::nullopt;
    }
    return make(state.arena.Make<ForeignProcPragma>(t.tokens(), *lang,
        result ? PredOrFunc::Function : PredOrFunc::Predicate, procName,
        state.arena.SaveArray(llvm::ArrayRef<const ForeignProcArg *>{procArgs}),
        result, *attributes,
        state.arena.Make<ForeignCode>(
            StripParentheses(arg(3)).tokens(), false, *code)));
  }
  if (name == "foreign_type") {
    if (!count(3, 4)) {
      return std::nullopt;
    }
    auto lang{Apply(language, arg(0), state)};
    auto head{Apply(typeHead, arg(1), state)};
    auto descriptor{
        Apply(expect("a foreign type descriptor string", stringLiteral), arg(2),
            state)};
    llvm::ArrayRef<ForeignTypeAssertion> assertions;
    bool ok{args.size() < 4 ||
        ConvertOptional(expect("a list of assertions",
                            listOf(EnumNameRule<ForeignTypeAssertion,
                                ForeignTypeAssertion_enumSize>{
                                "'can_pass_as_mercury_type', "
                                "'word_aligned_pointer' or 'stable'"})),
            &arg(3), assertions, state)};
    const TypeAttributes *attributes{nullptr};
    ok &= ConvertOptional(typeAttributes, where, attributes, state);
    if (!lang || !head || !descriptor || !ok) {
      return std::nullopt;
    }
    auto [typeName, parameters]{*head};
    return make(state.arena.Make<ForeignTypePragma>(t.tokens(), *lang,
        typeName, parameters, *descriptor, assertions, attributes));
  }
  if (name == "foreign_enum") {
    if (!count(3, 3)) {
      return std::nullopt;
    }
    auto lang{Apply(language, arg(0), state)};
    auto typeName{Apply(typeNameArity, arg(1), state)};
    auto values{Apply(enumValues, arg(2), state)};
    if (!lang || !typeName || !values) {
      return std::nullopt;
    }
    return make(state.arena.Make<ForeignEnumPragma>(
        t.tokens(), *lang, *typeName, *values));
  }
  if (name == "foreign_export_enum") {
    if (!count(2, 4)) {
      return std::nullopt;
    }
    auto lang{Apply(language, arg(0), state)};
    auto typeName{Apply(typeNameArity, arg(1), state)};
    std::optional<llvm::StringRef> prefix;
    bool uppercase{false}, ok{true};
    if (args.size() >= 3) {
      ok = ConvertExportEnumAttributes(arg(2), prefix, uppercase, state);
    }
    llvm::ArrayRef<const ForeignEnumValue *> overrides;
    if (args.size() == 4) {
      ok &= ConvertOptional(enumValues, &arg(3), overrides, state);
    }
    if (!lang || !typeName || !ok) {
      return std::nullopt;
    }
    return make(state.arena.Make<ForeignExportEnumPragma>(
        t.tokens(), *lang, *typeName, prefix, uppercase, overrides));
  }
  if (name == "foreign_export") {
    if (!count(3, 3)) {
      return std::nullopt;
    }
    auto lang{Apply(language, arg(0), state)};
    auto proc{Apply(procSpec, arg(1), state)};
    const Term &nameTerm{StripParentheses(arg(2))};
    const auto *foreignName{llvm::dyn_cast<StringTerm>(&nameTerm)};
    if (!foreignName || foreignName->value().empty()) {
      // mmc: "expected a non-empty string for the foreign language name".
      SayExpected(state, nameTerm, "the foreign name, a non-empty string");
    }
    if (!lang || !proc || !foreignName || foreignName->value().empty()) {
      return std::nullopt;
    }
    return make(state.arena.Make<ForeignExportPragma>(t.tokens(), *lang,
        *proc, state.arena.Save(foreignName->value())));
  }
  if (name == "foreign_import_module") {
    if (!count(2, 2)) {
      return std::nullopt;
    }
    auto lang{Apply(language, arg(0), state)};
    auto module{Apply(expect("a module name", symName), arg(1), state)};
    if (!lang || !module) {
      return std::nullopt;
    }
    return make(state.arena.Make<ForeignImportModulePragma>(
        t.tokens(), *lang, *module));
  }
  if (IsPragmaToDo(name)) {
    state.messages.Say(GetSource(t, state.tokens),
        "the ':- pragma %s' declaration"_todo_en_US, name.str());
    return std::nullopt;
  }
  // mmc: "`foo' is not a recognized pragma name".
  state.messages.Say(GetSource(*body, state.tokens),
      "'%s' is not a recognized pragma"_err_en_US, name.str());
  return std::nullopt;
}

} // namespace merkurius::parser
