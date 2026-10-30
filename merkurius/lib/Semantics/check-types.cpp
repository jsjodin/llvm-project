//===-- lib/Semantics/check-types.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Type checking, as the reference manual defines it: a clause is
// type-correct if it has a unique valid type assignment, up to renaming.
// The checker computes the set of them. It starts from one assignment,
// with the head's types from the declaration, and takes the goals in
// order: each goal keeps the assignments it is valid in, and extends them,
// once for each declaration that an overloaded name in it can refer to. A
// goal that leaves none is a type error, reported and then skipped, so
// that one clause can show several. In the end, assignments that chose
// different declarations for some name are an ambiguity.
//
// An assignment binds type variables, by number: the declaration's come
// first, and are rigid, since the head must match the declaration exactly
// (up to renaming); then one for each variable of the clause; then those
// of the clause's type qualifications and of the signatures instantiated
// for its goals, which unification binds.

#include "merkurius/Semantics/check-types.h"
#include "merkurius/Common/idioms.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

namespace merkurius::semantics {

using namespace parser::literals;

TypeChecking::TypeChecking() = default;
TypeChecking::~TypeChecking() = default;

std::string TypeChecking::Print(
    const ClauseTypes &clause, const Type &type) const {
  return TypeToString(type, [&](unsigned var) {
    return var < clause.typeVarNames.size() ? clause.typeVarNames[var].str()
                                            : "T_" + std::to_string(var);
  });
}

std::string SignatureToString(const Symbol &symbol, const Signature &sig) {
  std::string text;
  llvm::raw_string_ostream os{text};
  auto name{[&](unsigned var) {
    return var < sig.varNames.size() && !sig.varNames[var].empty()
        ? sig.varNames[var].str()
        : "T" + std::to_string(var);
  }};
  // The symbol printed as kind module.name/arity, without the arity.
  std::string printed;
  llvm::raw_string_ostream symbolText{printed};
  symbolText << symbol;
  os << llvm::StringRef{printed}.rsplit('/').first;
  bool function{sig.kind == parser::PredOrFunc::Function};
  llvm::ArrayRef<const Type *> arguments{sig.arguments};
  if (function) {
    arguments = arguments.drop_back();
  }
  if (!arguments.empty()) {
    os << '(';
    llvm::interleave(
        arguments, os, [&](const Type *t) { PrintType(os, *t, name); }, ", ");
    os << ')';
  }
  if (function) {
    os << " = ";
    PrintType(os, *sig.arguments.back(), name);
  }
  return text;
}

namespace {

// Type variable names in a declaration or a clause: each new name gets the
// next number, unless the names are closed.
struct TypeVarNames {
  llvm::StringMap<unsigned> numbers;
  std::vector<llvm::StringRef> names; // by number, for those named
  unsigned next{0};
  bool closed{false};

  std::optional<unsigned> Get(llvm::StringRef name) {
    if (auto it{numbers.find(name)}; it != numbers.end()) {
      return it->second;
    }
    if (closed) {
      return std::nullopt;
    }
    Add(name);
    return next - 1;
  }
  void Add(llvm::StringRef name) {
    numbers[name] = next;
    if (names.size() <= next) {
      names.resize(next + 1);
    }
    names[next++] = name;
  }
};

struct Assignment {
  std::vector<const Type *> bindings; // by type variable; null if unbound
  std::vector<const Symbol *> symbols; // by goal number
};

constexpr std::size_t maxAssignments{1000};

} // namespace

class TypeChecker {
public:
  TypeChecker(TypeChecking &result, const NameResolution &resolution,
      const parser::TokenSequence &tokens, parser::Messages &messages)
      : result_{result}, types_{result.types_}, resolution_{resolution},
        tokens_{tokens}, messages_{messages} {}

  void Run(const SemanticForm &);

private:
  // Declarations.
  const Type *Convert(const parser::Type &, TypeVarNames &);
  const Type *Expand(const Symbol &, llvm::ArrayRef<const Type *> arguments,
      parser::CharBlock where);
  const Signature *SignatureOf(const Symbol &);
  std::optional<Signature> MakeSignature(const Symbol &);
  const Type *Primitive(llvm::StringRef name);
  const Type *Library(llvm::StringRef module, llvm::StringRef name);
  bool IsError(const Type *type) const {
    return type->kind == TypeKind::Named && type->constructor == &error_;
  }

  // Clauses.
  void CheckClause(const ClauseForm &);
  void CheckGoal(const Goal &);
  void CheckUnify(const Unify &);
  void CheckFunctor(const Unify &, const parser::SymName &);
  void CheckCall(const Call &);
  void CheckHOCall(const HOCall &);
  void Finish(const ClauseForm &, const Symbol *definition,
      const Signature *signature);

  // Extends every assignment by each of `options`, as a goal does: each
  // option changes an assignment's copy, and returns whether it is valid.
  // If none is, `report` is called with the first assignment, which the
  // goal is then skipped for.
  using Option = std::function<bool(Assignment &)>;
  void Extend(llvm::ArrayRef<Option> options,
      llvm::function_ref<void(const Assignment &)> report);
  void Constrain(
      Option option, llvm::function_ref<void(const Assignment &)> report) {
    Extend({option}, report);
  }

  // Type operations within an assignment.
  const Type *VarType(VarId var) { return types_.Variable(rigid_ + var); }
  const Type *Walk(const Assignment &, const Type *) const;
  const Type *Resolve(const Assignment &, const Type *);
  bool UnifyTypes(Assignment &, const Type *, const Type *);
  bool Occurs(const Assignment &, unsigned var, const Type *) const;
  llvm::SmallVector<const Type *, 4> Instantiate(Assignment &, const Signature &);
  // Applies a signature of `symbol` to the arguments of a functor or call
  // whose value, if it is a functor, has type `value`.
  bool Apply(Assignment &, const Symbol &, const Signature &,
      llvm::ArrayRef<VarId> arguments, const Type *value);

  // Messages.
  // In the tokens of the module whose declaration or clause is checked.
  parser::CharBlock Source(TokenRange range) const {
    return {tokensNow_->tokens[range.first].spelling.begin(),
        tokensNow_->tokens[range.last].spelling.end()};
  }
  // Uses a declaration's tokens while it is in scope.
  struct InTokensOf {
    InTokensOf(TypeChecker &checker, const Symbol &symbol)
        : checker{checker}, saved{checker.tokensNow_} {
      if (symbol.tokens) {
        checker.tokensNow_ = symbol.tokens;
      }
    }
    ~InTokensOf() { checker.tokensNow_ = saved; }
    TypeChecker &checker;
    const parser::TokenSequence *saved;
  };
  std::string Describe(VarId) const;
  // Whether the variable is one the program names.
  bool IsNamed(VarId var) const {
    VarKind kind{clause_->vars[var].kind};
    return kind == VarKind::Program || kind == VarKind::State;
  }
  std::string TypeText(const Assignment &, const Type *);
  void AttachCandidates(parser::Message &, const parser::SymName &);

  TypeChecking &result_;
  TypeContext &types_;
  const NameResolution &resolution_;
  const parser::TokenSequence &tokens_;
  const parser::TokenSequence *tokensNow_{&tokens_};
  parser::Messages &messages_;
  // The type of an undefined name, which matches every type, so that it
  // draws no more errors.
  Symbol error_{SymbolKind::Type, "", "<error>", 0, std::monostate{}, {},
      false};
  llvm::DenseMap<const Symbol *, const Type *> equivalences_;
  llvm::DenseSet<const Symbol *> expanding_;

  // The clause being checked.
  const ClauseForm *clause_{nullptr};
  unsigned rigid_{0}; // the declaration's type variables
  llvm::ArrayRef<llvm::StringRef> rigidNames_;
  std::vector<Assignment> assignments_;
  std::vector<const parser::SymName *> names_; // by goal number
  bool tooMany_{false};
};

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

const Type *TypeChecker::Primitive(llvm::StringRef name) {
  for (const Symbol *symbol : resolution_.primitive().Lookup(name)) {
    return types_.Named(*symbol, {});
  }
  return types_.Named(error_, {});
}

// A type of a library module, if the module was loaded; null if not.
const Type *TypeChecker::Library(llvm::StringRef module, llvm::StringRef name) {
  if (const ModuleSymbols * symbols{resolution_.Module(module)}) {
    for (const Symbol *symbol : symbols->Lookup(name)) {
      if (symbol->kind == SymbolKind::Type && symbol->arity == 0) {
        return Expand(*symbol, {}, {});
      }
    }
  }
  return nullptr;
}

const Type *TypeChecker::Convert(
    const parser::Type &type, TypeVarNames &names) {
  return common::visit(
      common::visitors{
          [&](const parser::TypeVar *var) -> const Type * {
            if (std::optional<unsigned> number{names.Get(var->name)}) {
              return types_.Variable(*number);
            }
            messages_.Say(Source(var->tokens),
                "type variable %s is not a parameter of the type"_err_en_US,
                var->name.str());
            return types_.Named(error_, {});
          },
          [&](const parser::NamedType *named) -> const Type * {
            llvm::SmallVector<const Type *, 4> arguments;
            for (const parser::Type &argument : named->arguments) {
              arguments.push_back(Convert(argument, names));
            }
            // A type name can have several candidates, which must be one
            // type: char and char.char, for instance.
            llvm::SmallVector<const Type *, 2> found;
            for (const Symbol *candidate : named->name->candidates) {
              const Type *expanded{Expand(*candidate, arguments,
                  Source(named->name->tokens))};
              if (!llvm::is_contained(found, expanded)) {
                found.push_back(expanded);
              }
            }
            if (found.empty()) {
              return types_.Named(error_, {}); // undefined, and reported
            }
            if (found.size() > 1) {
              std::string list;
              llvm::interleave(
                  found,
                  [&](const Type *t) { list += "`" + TypeToString(*t) + "'"; },
                  [&] { list += ", "; });
              messages_.Say(Source(named->name->tokens),
                  "the type name %s is ambiguous: it may be %s"_err_en_US,
                  named->name->name.str(), list);
            }
            return found.front();
          },
          [&](const parser::TupleType *tuple) -> const Type * {
            llvm::SmallVector<const Type *, 4> elements;
            for (const parser::Type &element : tuple->elements) {
              elements.push_back(Convert(element, names));
            }
            return types_.Tuple(elements);
          },
          [&](const parser::HigherOrderType *higher) -> const Type * {
            llvm::SmallVector<const Type *, 4> arguments;
            for (const parser::TypeAndMode *argument : higher->arguments) {
              arguments.push_back(Convert(argument->type, names));
            }
            const Type *result{higher->result
                    ? Convert(higher->result->type, names)
                    : nullptr};
            return types_.HigherOrder(
                higher->kind, higher->purity, arguments, result);
          },
      },
      type);
}

// The type `symbol` applied to `arguments` names, with an equivalence
// expanded.
const Type *TypeChecker::Expand(const Symbol &symbol,
    llvm::ArrayRef<const Type *> arguments, parser::CharBlock where) {
  // mmc's character is char (see the design document's register).
  if (symbol.module.empty() && symbol.name == "character") {
    return Primitive("char");
  }
  const auto *defn{
      std::get_if<const parser::EquivalenceTypeDefn *>(&symbol.declaration)};
  if (!defn) {
    return types_.Named(symbol, arguments);
  }
  auto [it, inserted]{equivalences_.try_emplace(&symbol, nullptr)};
  if (inserted) {
    if (!expanding_.insert(&symbol).second) {
      equivalences_.erase(&symbol);
      return types_.Named(symbol, arguments);
    }
    InTokensOf in{*this, symbol};
    TypeVarNames names;
    for (llvm::StringRef parameter : (*defn)->parameters) {
      names.Add(parameter);
    }
    names.closed = true;
    const Type *body{Convert((*defn)->type, names)};
    expanding_.erase(&symbol);
    it = equivalences_.find(&symbol);
    it->second = body;
  } else if (!it->second) {
    // Expanding it reached it again.
    messages_.Say(where, "the equivalence type %s/%u is circular"_err_en_US,
        symbol.name.str(), symbol.arity);
    return types_.Named(error_, {});
  }
  return types_.Substitute(it->second, [&](unsigned var) {
    return var < arguments.size() ? arguments[var] : types_.Named(error_, {});
  });
}

const Signature *TypeChecker::SignatureOf(const Symbol &symbol) {
  auto it{result_.signatures_.find(&symbol)};
  if (it == result_.signatures_.end()) {
    std::optional<Signature> made{MakeSignature(symbol)};
    it = result_.signatures_.try_emplace(&symbol, made).first;
  }
  return it->second ? &*it->second : nullptr;
}

std::optional<Signature> TypeChecker::MakeSignature(const Symbol &symbol) {
  InTokensOf in{*this, symbol};
  TypeVarNames names;
  llvm::SmallVector<const Type *, 8> arguments;
  auto make{[&](parser::PredOrFunc kind, parser::Purity purity) {
    return Signature{kind, purity, names.next,
        result_.arena_.SaveArray(llvm::ArrayRef<llvm::StringRef>{names.names}),
        result_.arena_.SaveArray(llvm::ArrayRef<const Type *>{arguments})};
  }};
  if (const auto *decl{
          std::get_if<const parser::PredDecl *>(&symbol.declaration)}) {
    for (const parser::TypeAndMode *argument : (*decl)->arguments) {
      arguments.push_back(Convert(argument->type, names));
    }
    const Type *result{nullptr};
    if ((*decl)->withType) {
      // :- pred p(int) : foldl_pred(T, U) adds the type's arguments.
      const Type *type{Convert(*(*decl)->withType, names)};
      if (type->kind == TypeKind::HigherOrder && type->predOrFunc == (*decl)->kind) {
        arguments.append(type->arguments.begin(), type->arguments.end());
        result = type->result;
      } else if (!IsError(type)) {
        messages_.Say(Source((*decl)->tokens),
            "the type of %s is not a %s type"_err_en_US,
            (*decl)->name->name.str(),
            (*decl)->kind == parser::PredOrFunc::Function ? "func" : "pred");
      }
    }
    if ((*decl)->result) {
      result = Convert((*decl)->result->type, names);
    }
    if ((*decl)->kind == parser::PredOrFunc::Function) {
      arguments.push_back(result ? result : types_.Named(error_, {}));
    }
    return make((*decl)->kind, (*decl)->purity);
  }
  if (const auto *constructor{
          std::get_if<const parser::Constructor *>(&symbol.declaration)}) {
    const auto *defn{symbol.owner
            ? std::get_if<const parser::DuTypeDefn *>(&symbol.owner->declaration)
            : nullptr};
    if (!defn) {
      return std::nullopt;
    }
    llvm::SmallVector<const Type *, 4> parameters;
    for (llvm::StringRef parameter : (*defn)->parameters) {
      names.Add(parameter);
      parameters.push_back(types_.Variable(names.next - 1));
    }
    // Existential type variables are not yet told apart (design document).
    for (llvm::StringRef var : (*constructor)->existentialVars) {
      names.Add(var);
    }
    names.closed = true;
    const Type *owner{types_.Named(*symbol.owner, parameters)};
    if (symbol.kind == SymbolKind::Constructor) {
      for (const parser::ConstructorArg *argument : (*constructor)->arguments) {
        arguments.push_back(Convert(argument->type, names));
      }
      arguments.push_back(owner);
    } else {
      // f/1 is (T) = F, and 'f :='/2 is (T, F) = T.
      llvm::StringRef field{symbol.name};
      field.consume_back(" :=");
      const Type *type{types_.Named(error_, {})};
      for (const parser::ConstructorArg *argument : (*constructor)->arguments) {
        if (argument->field && argument->field->name == field) {
          type = Convert(argument->type, names);
        }
      }
      arguments.push_back(owner);
      arguments.push_back(type);
      if (symbol.kind == SymbolKind::FieldUpdate) {
        arguments.push_back(owner);
      }
    }
    return make(parser::PredOrFunc::Function, parser::Purity::Pure);
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Assignments
//===----------------------------------------------------------------------===//

const Type *TypeChecker::Walk(const Assignment &a, const Type *type) const {
  while (type->kind == TypeKind::Variable && type->var < a.bindings.size() &&
      a.bindings[type->var]) {
    type = a.bindings[type->var];
  }
  return type;
}

// The type with its bound variables replaced, throughout.
const Type *TypeChecker::Resolve(const Assignment &a, const Type *type) {
  return types_.Substitute(type, [&](unsigned var) {
    const Type *variable{types_.Variable(var)};
    const Type *bound{Walk(a, variable)};
    return bound == variable ? variable : Resolve(a, bound);
  });
}

bool TypeChecker::Occurs(
    const Assignment &a, unsigned var, const Type *type) const {
  type = Walk(a, type);
  if (type->kind == TypeKind::Variable) {
    return type->var == var;
  }
  return llvm::any_of(type->arguments,
             [&](const Type *t) { return Occurs(a, var, t); }) ||
      (type->result && Occurs(a, var, type->result));
}

bool TypeChecker::UnifyTypes(Assignment &a, const Type *x, const Type *y) {
  x = Walk(a, x);
  y = Walk(a, y);
  if (x == y || IsError(x) || IsError(y)) {
    return true;
  }
  for (auto [var, other] : {std::pair{x, y}, std::pair{y, x}}) {
    if (var->kind == TypeKind::Variable && var->var >= rigid_) {
      if (Occurs(a, var->var, other)) {
        return false;
      }
      a.bindings[var->var] = other;
      return true;
    }
  }
  if (x->kind != y->kind || x->constructor != y->constructor ||
      x->predOrFunc != y->predOrFunc || x->purity != y->purity ||
      x->arguments.size() != y->arguments.size() ||
      (x->result == nullptr) != (y->result == nullptr)) {
    return false; // two rigid variables among them
  }
  for (auto [left, right] : llvm::zip(x->arguments, y->arguments)) {
    if (!UnifyTypes(a, left, right)) {
      return false;
    }
  }
  return !x->result || UnifyTypes(a, x->result, y->result);
}

llvm::SmallVector<const Type *, 4> TypeChecker::Instantiate(
    Assignment &a, const Signature &sig) {
  unsigned offset(a.bindings.size());
  a.bindings.resize(offset + sig.typeVars);
  llvm::SmallVector<const Type *, 4> types;
  for (const Type *type : sig.arguments) {
    types.push_back(types_.Substitute(
        type, [&](unsigned var) { return types_.Variable(offset + var); }));
  }
  return types;
}

bool TypeChecker::Apply(Assignment &a, const Symbol &symbol,
    const Signature &sig, llvm::ArrayRef<VarId> arguments,
    const Type *value) {
  llvm::SmallVector<const Type *, 4> types{Instantiate(a, sig)};
  bool function{sig.kind == parser::PredOrFunc::Function};
  std::size_t arity{types.size() - (function ? 1 : 0)};
  if (arguments.size() > arity) {
    return false;
  }
  for (std::size_t j{0}; j < arguments.size(); ++j) {
    if (!UnifyTypes(a, VarType(arguments[j]), types[j])) {
      return false;
    }
  }
  if (!value) {
    return arguments.size() == arity; // a call
  }
  bool symbolIsFunction{symbol.kind != SymbolKind::Predicate};
  if (symbolIsFunction && arguments.size() == arity) {
    return UnifyTypes(a, value, types.back()); // an application
  }
  // A closure of the arguments not given (reference manual, "Creating
  // higher-order terms").
  llvm::ArrayRef<const Type *> rest{
      llvm::ArrayRef<const Type *>{types}.slice(arguments.size(),
          arity - arguments.size())};
  return UnifyTypes(a, value,
      types_.HigherOrder(sig.kind, sig.purity, rest,
          function ? types.back() : nullptr));
}

void TypeChecker::Extend(llvm::ArrayRef<Option> options,
    llvm::function_ref<void(const Assignment &)> report) {
  std::vector<Assignment> extended;
  for (const Assignment &a : assignments_) {
    for (const Option &option : options) {
      Assignment copy{a};
      if (option(copy)) {
        extended.push_back(std::move(copy));
      }
    }
  }
  if (extended.empty()) {
    report(assignments_.front());
    return;
  }
  if (extended.size() > maxAssignments) {
    if (!tooMany_) {
      messages_.Say(Source(clause_->tokens),
          "overloading is too ambiguous here: there are more than %zu possible type assignments; add type qualifications"_err_en_US,
          maxAssignments);
      tooMany_ = true;
    }
    extended.resize(maxAssignments);
  }
  assignments_ = std::move(extended);
}

//===----------------------------------------------------------------------===//
// Clauses
//===----------------------------------------------------------------------===//

std::string TypeChecker::Describe(VarId var) const {
  const VarInfo &info{clause_->vars[var]};
  switch (info.kind) {
  case VarKind::Program:
    return "variable " + info.name.str();
  case VarKind::State:
    return "state variable !" + info.name.str();
  default:
    return "`" + Source(info.tokens).ToString() + "'";
  }
}

std::string TypeChecker::TypeText(const Assignment &a, const Type *type) {
  return TypeToString(*Resolve(a, type), [&](unsigned var) {
    return var < rigid_ ? rigidNames_[var].str() : "T_" + std::to_string(var);
  });
}

void TypeChecker::AttachCandidates(
    parser::Message &message, const parser::SymName &name) {
  for (const Symbol *candidate : name.candidates) {
    if (const Signature * sig{SignatureOf(*candidate)}) {
      message.Attach(Source(name.tokens), "it may be %s"_because_en_US,
          SignatureToString(*candidate, *sig));
    }
  }
}

void TypeChecker::CheckUnify(const Unify &unify) {
  if (unify.isVar) {
    Constrain(
        [&](Assignment &a) {
          return UnifyTypes(a, VarType(unify.left), VarType(unify.var));
        },
        [&](const Assignment &a) {
          messages_.Say(Source(unify.tokens),
              "type error in unification of %s and %s: %s has type %s, but %s has type %s"_err_en_US,
              Describe(unify.left), Describe(unify.var), Describe(unify.left),
              TypeText(a, VarType(unify.left)), Describe(unify.var),
              TypeText(a, VarType(unify.var)));
        });
    return;
  }
  // The functor's type, when it does not depend on a declaration.
  const Type *type{nullptr};
  const Functor &functor{unify.functor};
  llvm::SmallVector<const Type *, 4> arguments;
  for (VarId argument : unify.arguments) {
    arguments.push_back(VarType(argument));
  }
  switch (functor.kind) {
  case FunctorKind::Name:
  case FunctorKind::Cons:
  case FunctorKind::Nil:
    if (const parser::SymName * name{resolution_.FunctorName(unify)}) {
      CheckFunctor(unify, *name);
    }
    return;
  case FunctorKind::Tuple:
    type = types_.Tuple(arguments);
    break;
  case FunctorKind::Literal: {
    llvm::StringRef text{functor.literal->text};
    if (functor.literal->kind == parser::LiteralKind::String) {
      type = Primitive("string");
    } else if (functor.literal->kind == parser::LiteralKind::Float) {
      type = Primitive("float");
    } else {
      // An integer's suffix names its type: 1i8 is an int8, 2u a uint.
      std::size_t at{text.find_first_of("iu")};
      llvm::StringRef suffix{
          at == llvm::StringRef::npos ? llvm::StringRef{} : text.substr(at)};
      std::string name{"int"};
      if (!suffix.empty()) {
        name = (suffix.front() == 'u' ? "uint" : "int") +
            suffix.drop_front().str();
      }
      type = Primitive(name);
    }
    break;
  }
  case FunctorKind::ImplementationLiteral:
    type = Primitive(functor.implementation->name == "line" ? "int" : "string");
    break;
  case FunctorKind::Coerce:
    messages_.Say(Source(unify.tokens),
        "type checking %s"_todo_en_US, "coerce expressions");
    return;
  case FunctorKind::Lambda: {
    const Lambda &lambda{*functor.lambda};
    llvm::SmallVector<const Type *, 4> types;
    for (VarId argument : lambda.arguments) {
      types.push_back(VarType(argument));
    }
    const Type *result{nullptr};
    if (lambda.kind == parser::PredOrFunc::Function && !types.empty()) {
      result = types.pop_back_val();
    }
    type = types_.HigherOrder(lambda.kind, lambda.purity, types, result);
    break;
  }
  }
  Constrain(
      [&](Assignment &a) { return UnifyTypes(a, VarType(unify.left), type); },
      [&](const Assignment &a) {
        std::string term{Source(unify.tokens).ToString()};
        if (IsNamed(unify.left)) {
          messages_.Say(Source(unify.tokens),
              "type error in unification of %s and `%s': %s has type %s, but `%s' has type %s"_err_en_US,
              Describe(unify.left), term, Describe(unify.left),
              TypeText(a, VarType(unify.left)), term, TypeText(a, type));
        } else {
          messages_.Say(Source(unify.tokens),
              "type error: `%s' has type %s, but must have type %s here"_err_en_US,
              term, TypeText(a, type), TypeText(a, VarType(unify.left)));
        }
      });
  if (functor.kind == FunctorKind::Lambda) {
    CheckGoal(functor.lambda->body);
  }
}

void TypeChecker::CheckFunctor(const Unify &unify, const parser::SymName &name) {
  names_[unify.number] = &name;
  std::vector<Option> options;
  for (const Symbol *candidate : name.candidates) {
    options.push_back([&, candidate](Assignment &a) {
      a.symbols[unify.number] = candidate;
      const Signature *sig{SignatureOf(*candidate)};
      // An undeclared predicate or function: its types are inferred later.
      return !sig ||
          Apply(a, *candidate, *sig, unify.arguments, VarType(unify.left));
    });
  }
  if (IsCharConstant(name, unify.arguments.size())) {
    options.push_back([&](Assignment &a) {
      return UnifyTypes(a, VarType(unify.left), Primitive("char"));
    });
  }
  if (options.empty()) {
    return; // undefined, and reported
  }
  Extend(options, [&](const Assignment &a) {
    names_[unify.number] = nullptr; // no declaration was chosen
    std::string types;
    for (VarId argument : unify.arguments) {
      types += (types.empty() ? "" : ", ") + TypeText(a, VarType(argument));
    }
    std::string term{Source(unify.tokens).ToString()};
    std::string symbol{name.name.str() + "/" +
        std::to_string(unify.arguments.size())};
    parser::Message &message{IsNamed(unify.left)
            ? messages_.Say(Source(unify.tokens),
                  "type error in unification of %s and `%s': %s has type %s, which no declaration of %s fits"_err_en_US,
                  Describe(unify.left), term, Describe(unify.left),
                  TypeText(a, VarType(unify.left)), symbol)
            : messages_.Say(Source(unify.tokens),
                  "type error: `%s' must have type %s here, which no declaration of %s fits"_err_en_US,
                  term, TypeText(a, VarType(unify.left)), symbol)};
    if (!types.empty()) {
      message.Attach(Source(unify.tokens),
          "its arguments have types %s"_because_en_US, types);
    }
    AttachCandidates(message, name);
  });
}

void TypeChecker::CheckCall(const Call &call) {
  if (call.event) {
    return;
  }
  names_[call.number] = call.name;
  std::vector<Option> options;
  for (const Symbol *candidate : call.name->candidates) {
    options.push_back([&, candidate](Assignment &a) {
      a.symbols[call.number] = candidate;
      const Signature *sig{SignatureOf(*candidate)};
      return !sig || Apply(a, *candidate, *sig, call.arguments, nullptr);
    });
  }
  if (options.empty()) {
    return;
  }
  Extend(options, [&](const Assignment &a) {
    names_[call.number] = nullptr;
    std::string types;
    for (VarId argument : call.arguments) {
      types += (types.empty() ? "" : ", ") + TypeText(a, VarType(argument));
    }
    parser::Message &message{messages_.Say(Source(call.tokens),
        "type error in call to %s/%zu: its arguments have types %s"_err_en_US,
        call.name->name.str(), call.arguments.size(), types)};
    AttachCandidates(message, *call.name);
  });
}

void TypeChecker::CheckHOCall(const HOCall &call) {
  bool function{call.kind == parser::PredOrFunc::Function};
  auto expected{[&](const Assignment &a) {
    llvm::SmallVector<const Type *, 4> types;
    for (VarId argument : call.arguments) {
      types.push_back(VarType(argument));
    }
    const Type *result{nullptr};
    if (function && !types.empty()) {
      result = types.pop_back_val();
    }
    // The call's purity is checked by purity checking, not here.
    const Type *closure{Walk(a, VarType(call.closure))};
    parser::Purity purity{closure->kind == TypeKind::HigherOrder
            ? closure->purity
            : parser::Purity::Pure};
    return types_.HigherOrder(call.kind, purity, types, result);
  }};
  Constrain(
      [&](Assignment &a) {
        return UnifyTypes(a, VarType(call.closure), expected(a));
      },
      [&](const Assignment &a) {
        messages_.Say(Source(call.tokens),
            "type error in higher-order call: %s has type %s, but the call needs %s"_err_en_US,
            Describe(call.closure), TypeText(a, VarType(call.closure)),
            TypeText(a, expected(a)));
      });
}

void TypeChecker::CheckGoal(const Goal &goal) {
  auto each{[&](llvm::ArrayRef<Goal> goals) {
    for (const Goal &g : goals) {
      CheckGoal(g);
    }
  }};
  common::visit(
      common::visitors{
          [&](const Conj *conj) { each(conj->goals); },
          [&](const Disj *disj) { each(disj->goals); },
          [&](const IfThenElse *ite) {
            CheckGoal(ite->condition);
            CheckGoal(ite->then);
            CheckGoal(ite->otherwise);
          },
          [&](const Not *negation) { CheckGoal(negation->goal); },
          [&](const Unify *unify) { CheckUnify(*unify); },
          [&](const Call *call) { CheckCall(*call); },
          [&](const HOCall *call) { CheckHOCall(*call); },
          [&](const Scope *scope) { CheckGoal(scope->goal); },
          [&](const Try *t) {
            if (const Type * io{Library("io", "state")}) {
              for (VarId var : t->io) {
                Constrain(
                    [&](Assignment &a) { return UnifyTypes(a, VarType(var), io); },
                    [&](const Assignment &a) {
                      messages_.Say(Source(t->tokens),
                          "type error: %s has type %s, but io(...) needs io.state"_err_en_US,
                          Describe(var), TypeText(a, VarType(var)));
                    });
              }
            }
            CheckGoal(t->goal);
            CheckGoal(t->then);
            if (t->otherwise) {
              CheckGoal(*t->otherwise);
            }
            for (const Catch &c : t->catches) {
              CheckGoal(c.pattern);
              CheckGoal(c.goal);
            }
            if (t->catchAny) {
              if (const Type * univ{Library("univ", "univ")}) {
                VarId var{t->catchAny->exception};
                Constrain(
                    [&](Assignment &a) { return UnifyTypes(a, VarType(var), univ); },
                    [&](const Assignment &a) {
                      messages_.Say(Source(t->catchAny->tokens),
                          "type error: %s has type %s, but catch_any needs univ.univ"_err_en_US,
                          Describe(var), TypeText(a, VarType(var)));
                    });
              }
              CheckGoal(t->catchAny->goal);
            }
          },
          [&](const Atomic *atomic) {
            // The states' types are stm_builtin's, which is not stubbed.
            for (const AtomicAlternative &alternative : atomic->alternatives) {
              CheckGoal(alternative.goal);
            }
          },
      },
      goal);
}

void TypeChecker::CheckClause(const ClauseForm &clause) {
  clause_ = &clause;
  tooMany_ = false;
  // What the clause defines, and its declared types, if any.
  const Symbol *definition{nullptr};
  SymbolKind kind{clause.kind == parser::PredOrFunc::Function
          ? SymbolKind::Function
          : SymbolKind::Predicate};
  for (const Symbol *candidate : clause.name->candidates) {
    if (candidate->kind == kind) {
      definition = candidate;
    }
  }
  const Signature *sig{definition ? SignatureOf(*definition) : nullptr};
  rigid_ = sig ? sig->typeVars : 0;
  rigidNames_ = sig ? sig->varNames : llvm::ArrayRef<llvm::StringRef>{};
  // The type qualifications' variables: the declaration's names are its
  // rigid variables, and the others are the clause's own.
  TypeVarNames names;
  for (unsigned var{0}; var < rigid_; ++var) {
    if (var < rigidNames_.size() && !rigidNames_[var].empty()) {
      names.numbers[rigidNames_[var]] = var;
    }
  }
  names.next = rigid_ + clause.vars.size();
  llvm::SmallVector<const Type *, 4> qualifications;
  for (const TypeQualification &q : clause.typeQualifications) {
    qualifications.push_back(Convert(q.type, names));
  }
  Assignment initial;
  initial.bindings.resize(names.next);
  initial.symbols.resize(clause.goals);
  assignments_ = {std::move(initial)};
  names_.assign(clause.goals, nullptr);
  if (sig) {
    for (std::size_t j{0}; j < clause.head.size() && j < sig->arguments.size();
         ++j) {
      Constrain(
          [&](Assignment &a) {
            return UnifyTypes(a, VarType(clause.head[j]), sig->arguments[j]);
          },
          [&](const Assignment &) {});
    }
  }
  for (std::size_t j{0}; j < qualifications.size(); ++j) {
    const TypeQualification &q{clause.typeQualifications[j]};
    const Type *type{qualifications[j]};
    Constrain([&](Assignment &a) { return UnifyTypes(a, VarType(q.var), type); },
        [&](const Assignment &a) {
          messages_.Say(Source(q.tokens),
              "type error: %s has type %s, but it is qualified as %s"_err_en_US,
              Describe(q.var), TypeText(a, VarType(q.var)), TypeText(a, type));
        });
  }
  CheckGoal(clause.body);
  Finish(clause, definition, sig);
}

void TypeChecker::Finish(const ClauseForm &clause, const Symbol *definition,
    const Signature *sig) {
  // Assignments that chose different declarations for a name are an
  // ambiguity; those that chose the same have the same types, since each
  // goal's unifications give its most general unifier.
  const Assignment &chosen{assignments_.front()};
  parser::Message *ambiguity{nullptr};
  for (unsigned goal{0}; goal < clause.goals; ++goal) {
    llvm::SmallVector<const Symbol *, 2> choices;
    for (const Assignment &a : assignments_) {
      if (!llvm::is_contained(choices, a.symbols[goal])) {
        choices.push_back(a.symbols[goal]);
      }
    }
    if (choices.size() < 2 || !names_[goal]) {
      continue;
    }
    if (!ambiguity) {
      // mmc: "ambiguous overloading causes type ambiguity".
      ambiguity = &messages_.Say(Source(clause.tokens),
          "ambiguous overloading causes type ambiguity"_err_en_US);
    }
    std::string list;
    llvm::raw_string_ostream os{list};
    llvm::interleave(
        choices, os,
        [&](const Symbol *s) {
          if (s) {
            os << *s;
          } else {
            os << "a char";
          }
        },
        " or ");
    ambiguity->Attach(Source(names_[goal]->tokens), "%s may be %s"_because_en_US,
        Source(names_[goal]->tokens).ToString(), list);
  }
  ClauseTypes types{&clause, definition, sig != nullptr, rigidNames_, {}, {},
      {}};
  std::vector<const Type *> vars;
  std::string unbound;
  for (VarId var{0}; var < clause.vars.size(); ++var) {
    const Type *type{Resolve(chosen, VarType(var))};
    vars.push_back(type);
    bool free{false};
    ForEachVariable(*type, [&](unsigned v) { free |= v >= rigid_; });
    if (free && sig && clause.vars[var].kind == VarKind::Program &&
        !ambiguity) {
      unbound += "\n  " + clause.vars[var].name.str() + ": " +
          TypeText(chosen, type);
    }
  }
  if (!unbound.empty()) {
    // mmc: "warning: unresolved polymorphism".
    messages_.Say(Source(clause.tokens),
        "unresolved polymorphism: the type variables of these types are not determined, and will be bound to void:%s"_warn_en_US,
        unbound);
  }
  types.vars = result_.arena_.SaveArray(llvm::ArrayRef<const Type *>{vars});
  types.symbols =
      result_.arena_.SaveArray(llvm::ArrayRef<const Symbol *>{chosen.symbols});
  types.names =
      result_.arena_.SaveArray(llvm::ArrayRef<const parser::SymName *>{names_});
  result_.clauses_.push_back(types);
}

void TypeChecker::Run(const SemanticForm &form) {
  for (const ClauseForm *clause : form.clauses) {
    CheckClause(*clause);
  }
}

std::unique_ptr<TypeChecking> CheckTypes(const SemanticForm &form,
    const NameResolution &resolution, const parser::TokenSequence &tokens,
    parser::Messages &messages) {
  auto result{std::make_unique<TypeChecking>()};
  TypeChecker{*result, resolution, tokens, messages}.Run(form);
  return result;
}

} // namespace merkurius::semantics
