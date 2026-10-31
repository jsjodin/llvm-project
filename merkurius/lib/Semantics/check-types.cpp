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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include <functional>
#include <set>

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

std::string TypeChecking::Print(
    const ClauseTypes &clause, const ClassConstraint &constraint) const {
  std::string text;
  llvm::raw_string_ostream os{text};
  PrintConstraint(os, constraint, [&](unsigned var) {
    return var < clause.typeVarNames.size() ? clause.typeVarNames[var].str()
                                            : "T_" + std::to_string(var);
  });
  return text;
}

void PrintConstraint(llvm::raw_ostream &os, const ClassConstraint &constraint,
    llvm::function_ref<std::string(unsigned)> name) {
  os << constraint.typeclass->module << '.' << constraint.typeclass->name
     << '(';
  llvm::interleave(
      constraint.arguments, os, [&](const Type *t) { PrintType(os, *t, name); },
      ", ");
  os << ')';
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
  if (!sig.constraints.empty()) {
    os << " <= (";
    llvm::interleave(
        sig.constraints, os,
        [&](const ClassConstraint &c) {
          PrintConstraint(os, c, name);
        },
        ", ");
    os << ')';
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
  // The constraints of the goals' callees, instantiated, by goal number.
  std::vector<std::pair<unsigned, ClassConstraint>> wanted;
};

bool Same(const ClassConstraint &x, const ClassConstraint &y) {
  // Types are interned.
  return x.typeclass == y.typeclass && x.arguments == y.arguments;
}

constexpr std::size_t maxAssignments{1000};
// The largest inferred argument type, as a tree.
constexpr std::size_t maxInferredSize{1000};

} // namespace

class TypeChecker {
public:
  TypeChecker(TypeChecking &result, const NameResolution &resolution,
      const parser::TokenSequence &tokens, parser::Messages &messages,
      const TypeCheckOptions &options)
      : result_{result}, types_{result.types_}, resolution_{resolution},
        tokens_{tokens}, messages_{&messages}, options_{options} {}

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
  bool HasError(const Type *type) const {
    return IsError(type) ||
        llvm::any_of(type->arguments, [&](const Type *t) { return HasError(t); }) ||
        (type->result && HasError(type->result));
  }

  // Type classes and instances (reference manual, "Type classes"). A
  // constraint whose type class is undefined or ambiguous is left out.
  std::optional<ClassConstraint> ConvertConstraint(
      const parser::ClassConstraint &, TypeVarNames &);
  // A type class's superclass constraints, on its parameters, which are
  // its type variables 0 to n-1; reported on if they break the rules.
  llvm::ArrayRef<ClassConstraint> Superclasses(const Symbol &);
  llvm::SmallVector<ClassConstraint, 2> SuperclassesOf(
      const parser::TypeclassDecl &, bool report);
  void CheckTypeclasses();
  void CheckInstances();
  void CheckMethods(const parser::InstanceDecl &, const Symbol &typeclass,
      const std::string &head);
  llvm::DenseMap<const Symbol *, llvm::ArrayRef<ClassConstraint>>
      superclasses_;
  // The instances that constraints may use: the loaded interfaces' and
  // the module's, without its errors and its abstract declarations.
  struct Instance {
    const InstanceItem *item;
    const Symbol *typeclass;
    llvm::ArrayRef<const Type *> types;
    llvm::ArrayRef<ClassConstraint> constraints;
  };
  std::vector<Instance> instances_;

  // Constraint satisfaction. The constraints assumed, those of a
  // declaration or an instance, with their superclasses, each with the
  // assumed one it comes from.
  using Assumed =
      std::vector<std::pair<ClassConstraint, const ClassConstraint *>>;
  Assumed Assume(llvm::ArrayRef<ClassConstraint>);
  // A proof of `wanted`, or the constraint it reduces to that has none.
  std::variant<ConstraintProof, ClassConstraint> Entail(
      const ClassConstraint &wanted, const Assumed &);
  llvm::SmallVector<ClassConstraint, 4> entailing_; // to stop at a cycle
  ClassConstraint SubstituteIn(
      const ClassConstraint &, llvm::function_ref<const Type *(unsigned)>);
  bool Match(const Type *pattern, const Type *type,
      llvm::SmallVectorImpl<const Type *> &bindings) const;
  std::string ConstraintText(const ClassConstraint &,
      llvm::function_ref<std::string(unsigned)> name);

  // Clauses. A clause is checked against its predicate's or function's
  // signature, declared or inferred, unless it is being inferred, when its
  // head's types are what it gives them.
  const Symbol *Definition(const ClauseForm &) const;
  ClauseTypes CheckClause(const ClauseForm &, bool inferring);
  void CheckGoal(const Goal &);
  void CheckUnify(const Unify &);
  void CheckFunctor(const Unify &, const parser::SymName &);
  void CheckCall(const Call &);
  void CheckHOCall(const HOCall &);
  ClauseTypes Finish(const ClauseForm &, const Symbol *definition,
      const Signature *signature);

  // Type inference for predicates and functions without declarations.
  using Undeclared =
      llvm::MapVector<const Symbol *, std::vector<const ClauseForm *>>;
  void Infer(const Undeclared &);
  // The signature that one pass of inference gives `symbol`'s clauses.
  // A pass's signature for a predicate or function, and whether its
  // clauses fit more than one, none of them more general than the others.
  struct Inferred {
    Signature signature;
    bool ambiguous;
  };
  Inferred InferPass(const Symbol &, llvm::ArrayRef<const ClauseForm *>);
  const Type *Generalization(const Type *, const Type *,
      std::map<std::pair<const Type *, const Type *>, unsigned> &);
  Signature General(const Symbol &, unsigned arity);
  Signature Generalize(
      const Symbol &, const Assignment &, llvm::ArrayRef<const Type *>);
  void ForEachReference(
      const Goal &, llvm::function_ref<void(const Symbol &)>) const;

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
  bool UnifyParts(Assignment &, const Type *, const Type *);
  llvm::DenseSet<std::pair<const Type *, const Type *>> *unified_{nullptr};
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
        : InTokensOf{checker, symbol.tokens} {}
    InTokensOf(TypeChecker &checker, const parser::TokenSequence *tokens)
        : checker{checker}, saved{checker.tokensNow_} {
      if (tokens) {
        checker.tokensNow_ = tokens;
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
  parser::Messages *messages_; // a scratch list while inferring
  const TypeCheckOptions &options_;
  Symbol &error_{result_.error_};
  llvm::DenseMap<const Symbol *, const Type *> equivalences_;
  llvm::DenseSet<const Symbol *> expanding_;

  // The clause being checked.
  const ClauseForm *clause_{nullptr};
  unsigned rigid_{0}; // the declaration's type variables
  llvm::ArrayRef<llvm::StringRef> rigidNames_;
  std::vector<Assignment> assignments_;
  std::vector<const parser::SymName *> names_; // by goal number
  unsigned goal_{0}; // the goal whose options are tried
  bool tooMany_{false};
  // The clause has a type error, or a name that is undefined; an ambiguity
  // or unresolved polymorphism may then be only its consequence, and is
  // not reported.
  bool failed_{false};
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
            messages_->Say(Source(var->tokens),
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
              messages_->Say(Source(named->name->tokens),
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
    messages_->Say(where, "the equivalence type %s/%u is circular"_err_en_US,
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
  llvm::SmallVector<ClassConstraint, 2> constraints;
  auto make{[&](parser::PredOrFunc kind, parser::Purity purity) {
    return Signature{kind, purity, names.next,
        result_.arena_.SaveArray(llvm::ArrayRef<llvm::StringRef>{names.names}),
        result_.arena_.SaveArray(llvm::ArrayRef<const Type *>{arguments}),
        result_.arena_.SaveArray(
            llvm::ArrayRef<ClassConstraint>{constraints})};
  }};
  if (const auto *decl{
          std::get_if<const parser::PredDecl *>(&symbol.declaration)}) {
    const auto *typeclass{symbol.owner
            ? std::get_if<const parser::TypeclassDecl *>(
                  &symbol.owner->declaration)
            : nullptr};
    if (typeclass) {
      llvm::SmallVector<const Type *, 4> parameters;
      for (llvm::StringRef parameter : (*typeclass)->parameters) {
        names.Add(parameter);
        parameters.push_back(types_.Variable(names.next - 1));
      }
      constraints.push_back({symbol.owner,
          result_.arena_.SaveArray(
              llvm::ArrayRef<const Type *>{parameters})});
    }
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
      }
      // Otherwise name resolution reported it.
    }
    if ((*decl)->result) {
      result = Convert((*decl)->result->type, names);
    }
    if ((*decl)->kind == parser::PredOrFunc::Function) {
      arguments.push_back(result ? result : types_.Named(error_, {}));
    }
    for (const parser::Constraint &constraint :
        (*decl)->universalConstraints) {
      if (const auto *c{
              std::get_if<const parser::ClassConstraint *>(&constraint)}) {
        if (std::optional<ClassConstraint> converted{
                ConvertConstraint(**c, names)}) {
          constraints.push_back(*converted);
        }
      }
    }
    // Each variable of a constraint must be determined by the argument
    // types (reference manual, "Type class constraints on predicates and
    // functions"); a method's, its type class's parameters too.
    llvm::DenseSet<unsigned> determined;
    for (const Type *argument : arguments) {
      ForEachVariable(*argument, [&](unsigned var) { determined.insert(var); });
    }
    llvm::DenseSet<unsigned> reported;
    bool function{(*decl)->kind == parser::PredOrFunc::Function};
    for (const ClassConstraint &constraint : constraints) {
      for (const Type *argument : constraint.arguments) {
        ForEachVariable(*argument, [&](unsigned var) {
          if (!determined.contains(var) && reported.insert(var).second) {
            messages_->Say(Source((*decl)->tokens),
                "type variable %s occurs in the constraints of %s %s/%u, but not in its %s"_err_en_US,
                names.names[var].str(), function ? "function" : "predicate",
                symbol.name.str(), symbol.arity,
                function ? "argument or result types" : "argument types");
          }
        });
      }
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
  // Each variable's binding is resolved once.
  llvm::DenseMap<unsigned, const Type *> resolved;
  std::function<const Type *(const Type *)> resolve{[&](const Type *t) {
    return types_.Substitute(t, [&](unsigned var) {
      if (auto it{resolved.find(var)}; it != resolved.end()) {
        return it->second;
      }
      const Type *variable{types_.Variable(var)};
      const Type *bound{Walk(a, variable)};
      const Type *result{bound == variable ? variable : resolve(bound)};
      resolved[var] = result;
      return result;
    });
  }};
  return resolve(type);
}

bool TypeChecker::Occurs(
    const Assignment &a, unsigned var, const Type *type) const {
  // Each shared part once (see types.cpp).
  llvm::DenseSet<const Type *> seen;
  std::function<bool(const Type *)> occurs{[&](const Type *t) {
    t = Walk(a, t);
    if (t->kind == TypeKind::Variable) {
      return t->var == var;
    }
    if (!seen.insert(t).second) {
      return false;
    }
    return llvm::any_of(t->arguments, occurs) ||
        (t->result && occurs(t->result));
  }};
  return occurs(type);
}

bool TypeChecker::UnifyTypes(Assignment &a, const Type *x, const Type *y) {
  // A pair of shared parts is unified once (see types.cpp).
  llvm::DenseSet<std::pair<const Type *, const Type *>> unified;
  unified_ = &unified;
  bool result{UnifyParts(a, x, y)};
  unified_ = nullptr;
  return result;
}

bool TypeChecker::UnifyParts(Assignment &a, const Type *x, const Type *y) {
  x = Walk(a, x);
  y = Walk(a, y);
  if (x == y || IsError(x) || IsError(y)) {
    return true;
  }
  if (x->kind != TypeKind::Variable && y->kind != TypeKind::Variable &&
      !unified_->insert({x, y}).second) {
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
    if (!UnifyParts(a, left, right)) {
      return false;
    }
  }
  return !x->result || UnifyParts(a, x->result, y->result);
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
  unsigned offset(a.bindings.size());
  llvm::SmallVector<const Type *, 4> types{Instantiate(a, sig)};
  for (const ClassConstraint &constraint : sig.constraints) {
    a.wanted.emplace_back(goal_, SubstituteIn(constraint, [&](unsigned var) {
      return types_.Variable(offset + var);
    }));
  }
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
    failed_ = true;
    report(assignments_.front());
    return;
  }
  if (extended.size() > maxAssignments) {
    if (!tooMany_) {
      messages_->Say(Source(clause_->tokens),
          "overloading is too ambiguous here: there are more than %zu possible type assignments; add type qualifications"_err_en_US,
          maxAssignments);
      tooMany_ = true;
    }
    extended.resize(maxAssignments);
  }
  assignments_ = std::move(extended);
}

//===----------------------------------------------------------------------===//
// Type classes and instances
//===----------------------------------------------------------------------===//

namespace {

// Whether a constraint's argument is a type variable or a ground type, as
// those of a type class's or an instance's constraints must be.
bool VariableOrGround(const Type &type) {
  if (type.kind == TypeKind::Variable) {
    return true;
  }
  bool ground{true};
  ForEachVariable(type, [&](unsigned) { ground = false; });
  return ground;
}

// Constraints as a set, to compare those of two declarations whose type
// variables are numbered alike.
using ConstraintSet =
    std::set<std::pair<const Symbol *, std::vector<const Type *>>>;
ConstraintSet AsSet(llvm::ArrayRef<ClassConstraint> constraints) {
  ConstraintSet set;
  for (const ClassConstraint &c : constraints) {
    set.emplace(c.typeclass,
        std::vector<const Type *>{c.arguments.begin(), c.arguments.end()});
  }
  return set;
}

std::string KindName(SymbolKind kind) {
  return kind == SymbolKind::Function ? "function" : "predicate";
}

} // namespace

std::optional<ClassConstraint> TypeChecker::ConvertConstraint(
    const parser::ClassConstraint &constraint, TypeVarNames &names) {
  llvm::SmallVector<const Type *, 4> arguments;
  for (const parser::Type &argument : constraint.arguments) {
    arguments.push_back(Convert(argument, names));
  }
  if (constraint.name->candidates.size() != 1) {
    return std::nullopt; // reported by name resolution
  }
  return ClassConstraint{constraint.name->candidates.front(),
      result_.arena_.SaveArray(llvm::ArrayRef<const Type *>{arguments})};
}

llvm::SmallVector<ClassConstraint, 2> TypeChecker::SuperclassesOf(
    const parser::TypeclassDecl &decl, bool report) {
  llvm::SmallVector<ClassConstraint, 2> supers;
  for (const parser::TypeclassConstraint &item : decl.constraints) {
    const auto *constraint{
        std::get_if<const parser::ClassConstraint *>(&item)};
    if (!constraint) {
      continue; // a functional dependency
    }
    TypeVarNames names;
    for (llvm::StringRef parameter : decl.parameters) {
      names.Add(parameter);
    }
    std::optional<ClassConstraint> converted{
        ConvertConstraint(**constraint, names)};
    if (!converted) {
      continue;
    }
    bool valid{true};
    for (unsigned var{unsigned(decl.parameters.size())}; var < names.next;
         ++var) {
      if (report) {
        messages_->Say(Source((*constraint)->tokens),
            "type variable %s in the superclass constraint is not a parameter of the type class"_err_en_US,
            names.names[var].str());
      }
      valid = false;
    }
    if (!llvm::all_of(converted->arguments,
            [](const Type *t) { return VariableOrGround(*t); })) {
      if (report) {
        messages_->Say(Source((*constraint)->tokens),
            "a superclass constraint may only constrain type variables and ground types"_err_en_US);
      }
      valid = false;
    }
    if (valid) {
      supers.push_back(*converted);
    }
  }
  return supers;
}

llvm::ArrayRef<ClassConstraint> TypeChecker::Superclasses(
    const Symbol &typeclass) {
  if (auto it{superclasses_.find(&typeclass)}; it != superclasses_.end()) {
    return it->second;
  }
  llvm::SmallVector<ClassConstraint, 2> supers;
  if (const auto *decl{
          std::get_if<const parser::TypeclassDecl *>(&typeclass.declaration)}) {
    InTokensOf in{*this, typeclass};
    supers = SuperclassesOf(**decl, true);
  }
  llvm::ArrayRef<ClassConstraint> saved{
      result_.arena_.SaveArray(llvm::ArrayRef<ClassConstraint>{supers})};
  superclasses_[&typeclass] = saved;
  return saved;
}

// The rules of the reference manual's "Typeclass declarations", "Abstract
// typeclass declarations" and "Type class constraints on type class
// declarations"; the parser checks that the parameters are distinct
// variables.
void TypeChecker::CheckTypeclasses() {
  llvm::MapVector<const Symbol *, llvm::SmallVector<const TypeclassItem *, 2>>
      bySymbol;
  for (const TypeclassItem &item : resolution_.typeclasses()) {
    bySymbol[item.symbol].push_back(&item);
  }
  auto where{[&](const parser::TypeclassDecl &decl) {
    return Source(decl.name->tokens);
  }};
  for (const auto &[symbol, items] : bySymbol) {
    (void)Superclasses(*symbol); // for its errors
    const TypeclassItem *definition{nullptr}, *abstract{nullptr};
    for (const TypeclassItem *item : items) {
      if (item->decl->abstract) {
        abstract = abstract ? abstract : item;
      } else if (definition) {
        parser::Message &message{messages_->Say(where(*item->decl),
            "typeclass %s/%u is defined more than once"_err_en_US,
            symbol->name.str(), symbol->arity)};
        message.Attach(where(*definition->decl),
            "it is first defined here"_because_en_US);
      } else {
        definition = item;
      }
    }
    if (abstract && !definition) {
      messages_->Say(where(*abstract->decl),
          "the abstract typeclass %s/%u has no definition"_err_en_US,
          symbol->name.str(), symbol->arity);
    } else if (abstract) {
      // The parameters are numbered by position in both.
      if (AsSet(SuperclassesOf(*abstract->decl, false)) !=
          AsSet(Superclasses(*symbol))) {
        parser::Message &message{messages_->Say(where(*definition->decl),
            "the superclass constraints of typeclass %s/%u differ from those of its abstract declaration"_err_en_US,
            symbol->name.str(), symbol->arity)};
        message.Attach(where(*abstract->decl),
            "the abstract declaration is here"_because_en_US);
      }
    }
  }
  // The superclass relation must be acyclic: each cycle is reported once,
  // at the first of its type classes that the module declares.
  llvm::DenseSet<const Symbol *> inReportedCycle;
  for (const auto &[symbol, items] : bySymbol) {
    if (inReportedCycle.contains(symbol)) {
      continue;
    }
    std::vector<const Symbol *> path{symbol};
    llvm::DenseSet<const Symbol *> visited;
    std::function<bool(const Symbol &)> reaches{[&](const Symbol &from) {
      for (const ClassConstraint &super : Superclasses(from)) {
        if (super.typeclass == symbol) {
          return true;
        }
        if (visited.insert(super.typeclass).second) {
          path.push_back(super.typeclass);
          if (reaches(*super.typeclass)) {
            return true;
          }
          path.pop_back();
        }
      }
      return false;
    }};
    if (!reaches(*symbol)) {
      continue;
    }
    std::string cycle;
    for (const Symbol *s : path) {
      inReportedCycle.insert(s);
      cycle += s->module.str() + "." + s->name.str() + "/" +
          std::to_string(s->arity) + " <= ";
    }
    cycle += symbol->module.str() + "." + symbol->name.str() + "/" +
        std::to_string(symbol->arity);
    const TypeclassItem &first{*items.front()};
    messages_->Say(where(*first.decl),
        "the superclass relation is cyclic: %s"_err_en_US, cycle);
  }
}

// The rules of the reference manual's "Instance declarations", "Abstract
// instance declarations" and "Type class constraints on instance
// declarations". Whether the instance's superclass constraints are
// satisfied is checked with the constraints of clauses.
void TypeChecker::CheckInstances() {
  // An instance's type class and its types' constructors, by position: a
  // tuple's arity stands for its constructor. Since each type's arguments
  // are variables, two instances overlap if their keys are equal.
  using Key = std::vector<std::uintptr_t>;
  struct Head {
    const InstanceItem *item;
    const Symbol *typeclass;
    llvm::SmallVector<const Type *, 4> types;
    llvm::SmallVector<ClassConstraint, 2> constraints;
    std::string text;
    Key key;
    bool valid;
    std::vector<llvm::StringRef> names; // of its type variables
  };
  std::vector<Head> heads;
  for (const InstanceItem &item : resolution_.instances()) {
    const parser::InstanceDecl &decl{*item.decl};
    if (decl.name->candidates.size() != 1) {
      continue; // reported by name resolution
    }
    InTokensOf in{*this, item.tokens};
    Head head{&item, decl.name->candidates.front(), {}, {}, {}, {}, true, {}};
    TypeVarNames names;
    for (const parser::Type &type : decl.types) {
      head.types.push_back(Convert(type, names));
    }
    auto name{[&](unsigned var) {
      return var < names.names.size() ? names.names[var].str()
                                      : "T" + std::to_string(var);
    }};
    {
      llvm::raw_string_ostream os{head.text};
      os << head.typeclass->module << '.' << head.typeclass->name << '(';
      llvm::interleave(
          head.types, os, [&](const Type *t) { PrintType(os, *t, name); },
          ", ");
      os << ')';
    }
    head.key.push_back(reinterpret_cast<std::uintptr_t>(head.typeclass));
    unsigned headVars{names.next};
    unsigned position{0};
    for (const Type *type : head.types) {
      ++position;
      if (IsError(type)) {
        head.valid = false;
        continue;
      }
      if (type->kind == TypeKind::Variable ||
          type->kind == TypeKind::HigherOrder) {
        if (item.local) {
          messages_->Say(Source(decl.name->tokens),
              "in instance %s, type %u, `%s', is a %s; it must be a type constructor applied to type variables"_err_en_US,
              head.text, position, TypeToString(*type, name),
              type->kind == TypeKind::Variable ? "type variable"
                                               : "higher-order type");
        }
        head.valid = false;
        continue;
      }
      head.key.push_back(type->kind == TypeKind::Tuple
              ? type->arguments.size()
              : reinterpret_cast<std::uintptr_t>(type->constructor));
      for (unsigned i{0}; i < type->arguments.size(); ++i) {
        if (type->arguments[i]->kind != TypeKind::Variable) {
          if (item.local) {
            messages_->Say(Source(decl.name->tokens),
                "in instance %s, argument %u of type %u, `%s', is not a type variable"_err_en_US,
                head.text, i + 1, position,
                TypeToString(*type->arguments[i], name));
          }
          head.valid = false;
          break;
        }
      }
    }
    for (const parser::ClassConstraint *constraint : decl.constraints) {
      unsigned before{names.next};
      std::optional<ClassConstraint> converted{
          ConvertConstraint(*constraint, names)};
      if (item.local) {
        for (unsigned var{std::max(before, headVars)}; var < names.next;
             ++var) {
          messages_->Say(Source(constraint->tokens),
              "type variable %s in the constraints of instance %s does not occur in its types"_err_en_US,
              names.names[var].str(), head.text);
        }
        if (converted &&
            !llvm::all_of(converted->arguments,
                [](const Type *t) { return VariableOrGround(*t); })) {
          messages_->Say(Source(constraint->tokens),
              "an instance's constraint may only constrain type variables and ground types"_err_en_US);
        }
      }
      if (names.next > headVars) {
        head.valid = false; // a variable not in the types
      }
      if (converted) {
        head.constraints.push_back(*converted);
      }
    }
    head.names = names.names;
    if (item.local && !decl.abstract && item.exported) {
      messages_->Say(Source(decl.name->tokens),
          "the instance %s is not abstract, so it may not be in the interface section"_err_en_US,
          head.text);
    }
    heads.push_back(std::move(head));
  }
  // Overlapping instances: the module's against the loaded interfaces'
  // and its own earlier ones.
  std::map<Key, const Head *> seen;
  for (const Head &head : heads) {
    if (!head.valid || (head.item->local && head.item->decl->abstract)) {
      continue;
    }
    auto [it, inserted]{seen.try_emplace(head.key, &head)};
    if (!inserted && head.item->local) {
      InTokensOf in{*this, head.item->tokens};
      parser::Message &message{messages_->Say(
          Source(head.item->decl->name->tokens),
          "instance %s overlaps an earlier instance of typeclass %s/%u"_err_en_US,
          head.text, head.typeclass->name.str(), head.typeclass->arity)};
      InTokensOf other{*this, it->second->item->tokens};
      message.Attach(Source(it->second->item->decl->name->tokens),
          "the earlier instance is %s"_because_en_US, it->second->text);
    }
  }
  for (const Head &head : heads) {
    if (head.valid && !(head.item->local && head.item->decl->abstract)) {
      instances_.push_back({head.item, head.typeclass,
          result_.arena_.SaveArray(llvm::ArrayRef<const Type *>{head.types}),
          result_.arena_.SaveArray(
              llvm::ArrayRef<ClassConstraint>{head.constraints})});
    }
  }
  for (const Head &head : heads) {
    if (!head.item->local || !head.valid) {
      continue;
    }
    const parser::InstanceDecl &decl{*head.item->decl};
    if (!decl.abstract) {
      CheckMethods(decl, *head.typeclass, head.text);
      // Its type class's superclass constraints, on its types, must follow
      // from its constraints and the instances (thesis, section 3.1.4).
      Assumed assumed{Assume(head.constraints)};
      for (const ClassConstraint &super : Superclasses(*head.typeclass)) {
        ClassConstraint wanted{SubstituteIn(super, [&](unsigned var) {
          return var < head.types.size() ? head.types[var]
                                         : types_.Named(error_, {});
        })};
        if (std::holds_alternative<ClassConstraint>(Entail(wanted, assumed))) {
          messages_->Say(Source(decl.name->tokens),
              "the superclass constraint %s of instance %s is not satisfied"_err_en_US,
              ConstraintText(wanted,
                  [&](unsigned var) {
                    return var < head.names.size() ? head.names[var].str()
                                                   : "T" + std::to_string(var);
                  }),
              head.text);
        }
      }
      continue;
    }
    // An abstract instance needs a definition with the same types and
    // constraints, up to renaming: variables are numbered in the order
    // they occur, so the types and constraints compare as they are.
    const Head *definition{nullptr};
    for (const Head &other : heads) {
      if (other.item->local && !other.item->decl->abstract &&
          other.typeclass == head.typeclass && other.types == head.types) {
        definition = &other;
        break;
      }
    }
    if (!definition) {
      messages_->Say(Source(decl.name->tokens),
          "the abstract instance %s has no definition in the implementation section"_err_en_US,
          head.text);
      continue;
    }
    if (AsSet(head.constraints) != AsSet(definition->constraints)) {
      parser::Message &message{
          messages_->Say(Source(definition->item->decl->name->tokens),
              "the constraints of instance %s differ from those of its abstract declaration"_err_en_US,
              definition->text)};
      message.Attach(
          Source(decl.name->tokens), "the abstract declaration is here"_because_en_US);
    }
  }
}

// Each method must be defined once, by clauses or by naming a predicate
// or function, and each definition must be of a method.
void TypeChecker::CheckMethods(const parser::InstanceDecl &decl,
    const Symbol &typeclass, const std::string &head) {
  const auto *classDecl{
      std::get_if<const parser::TypeclassDecl *>(&typeclass.declaration)};
  if (!classDecl || (*classDecl)->abstract) {
    messages_->Say(Source(decl.name->tokens),
        "the methods of typeclass %s/%u are not visible here, where its declaration is abstract"_err_en_US,
        typeclass.name.str(), typeclass.arity);
    return;
  }
  const ModuleSymbols *symbols{typeclass.module == resolution_.module().name()
          ? &resolution_.module()
          : resolution_.Module(typeclass.module)};
  std::vector<const Symbol *> methods;
  if (symbols) {
    for (const Symbol *symbol : symbols->symbols()) {
      if (symbol->owner == &typeclass) {
        methods.push_back(symbol);
      }
    }
  }
  // The first definition of each method, and whether it names one.
  llvm::DenseMap<const Symbol *, std::pair<TokenRange, bool>> defined;
  for (const parser::InstanceMethod &method : decl.methods) {
    SymbolKind kind{SymbolKind::Predicate};
    llvm::StringRef name;
    unsigned arity{0};
    TokenRange tokens;
    bool byName{false};
    common::visit(
        common::visitors{
            [&](const parser::MethodByName *m) {
              kind = m->kind == parser::PredOrFunc::Function
                  ? SymbolKind::Function
                  : SymbolKind::Predicate;
              name = m->method->name->name;
              arity = m->method->arity;
              tokens = m->tokens;
              byName = true;
            },
            [&](const parser::FactOrRule *clause) {
              kind = clause->kind == parser::PredOrFunc::Function
                  ? SymbolKind::Function
                  : SymbolKind::Predicate;
              name = clause->name->name;
              arity = clause->arguments.size();
              tokens = clause->name->tokens;
            },
            [&](const parser::DcgRule *rule) {
              name = rule->name->name;
              arity = rule->arguments.size() + 2;
              tokens = rule->name->tokens;
            },
        },
        method);
    auto found{llvm::find_if(methods, [&](const Symbol *m) {
      return m->kind == kind && m->name == name && m->arity == arity;
    })};
    if (found == methods.end()) {
      messages_->Say(Source(tokens),
          "the typeclass %s/%u has no %s method %s/%u"_err_en_US,
          typeclass.name.str(), typeclass.arity, KindName(kind), name.str(),
          arity);
      continue;
    }
    auto [it, inserted]{defined.try_emplace(*found, tokens, byName)};
    // Several clauses are one definition.
    if (!inserted && (byName || it->second.second)) {
      parser::Message &message{messages_->Say(Source(tokens),
          "instance %s has more than one implementation of the %s method %s/%u"_err_en_US,
          head, KindName(kind), name.str(), arity)};
      message.Attach(Source(it->second.first),
          "the first is here"_because_en_US);
    }
  }
  for (const Symbol *method : methods) {
    if (!defined.contains(method)) {
      messages_->Say(Source(decl.name->tokens),
          "instance %s has no implementation of the %s method %s/%u"_err_en_US,
          head, KindName(method->kind), method->name.str(), method->arity);
    }
  }
}

ClassConstraint TypeChecker::SubstituteIn(const ClassConstraint &constraint,
    llvm::function_ref<const Type *(unsigned)> replace) {
  llvm::SmallVector<const Type *, 4> arguments;
  for (const Type *argument : constraint.arguments) {
    arguments.push_back(types_.Substitute(argument, replace));
  }
  return {constraint.typeclass,
      result_.arena_.SaveArray(llvm::ArrayRef<const Type *>{arguments})};
}

std::string TypeChecker::ConstraintText(const ClassConstraint &constraint,
    llvm::function_ref<std::string(unsigned)> name) {
  std::string text;
  llvm::raw_string_ostream os{text};
  os << '`';
  PrintConstraint(os, constraint, name);
  os << '\'';
  return text;
}

TypeChecker::Assumed TypeChecker::Assume(
    llvm::ArrayRef<ClassConstraint> constraints) {
  Assumed assumed;
  for (const ClassConstraint &constraint : constraints) {
    assumed.emplace_back(constraint, &constraint);
  }
  // A constraint assumes its type class's superclasses, on its arguments.
  for (std::size_t i{0}; i < assumed.size(); ++i) {
    auto [constraint, origin]{assumed[i]};
    for (const ClassConstraint &super : Superclasses(*constraint.typeclass)) {
      ClassConstraint implied{SubstituteIn(super, [&](unsigned var) {
        return var < constraint.arguments.size() ? constraint.arguments[var]
                                                 : types_.Named(error_, {});
      })};
      if (llvm::none_of(assumed,
              [&](const auto &a) { return Same(a.first, implied); })) {
        assumed.emplace_back(implied, origin);
      }
    }
  }
  return assumed;
}

// Whether `type` is an instance of `pattern`, an instance's type, whose
// variables `bindings` binds.
bool TypeChecker::Match(const Type *pattern, const Type *type,
    llvm::SmallVectorImpl<const Type *> &bindings) const {
  if (pattern->kind == TypeKind::Variable) {
    if (bindings.size() <= pattern->var) {
      bindings.resize(pattern->var + 1, nullptr);
    }
    const Type *&bound{bindings[pattern->var]};
    if (!bound) {
      bound = type;
    }
    return bound == type;
  }
  if (type->kind != pattern->kind || type->constructor != pattern->constructor ||
      type->arguments.size() != pattern->arguments.size() ||
      (type->result == nullptr) != (pattern->result == nullptr)) {
    return false;
  }
  for (auto [p, t] : llvm::zip(pattern->arguments, type->arguments)) {
    if (!Match(p, t, bindings)) {
      return false;
    }
  }
  return !pattern->result || Match(pattern->result, type->result, bindings);
}

std::variant<ConstraintProof, ClassConstraint> TypeChecker::Entail(
    const ClassConstraint &wanted, const Assumed &assumed) {
  for (const auto &[constraint, origin] : assumed) {
    if (Same(constraint, wanted)) {
      return ConstraintProof{wanted, origin, nullptr, {}};
    }
  }
  if (llvm::any_of(entailing_,
          [&](const ClassConstraint &c) { return Same(c, wanted); })) {
    return wanted; // it needs itself
  }
  // Instances do not overlap, so at most one matches.
  for (const Instance &instance : instances_) {
    llvm::SmallVector<const Type *, 4> bindings;
    if (instance.typeclass != wanted.typeclass ||
        !llvm::all_of(llvm::zip(instance.types, wanted.arguments),
            [&](auto pair) {
              return Match(std::get<0>(pair), std::get<1>(pair), bindings);
            })) {
      continue;
    }
    entailing_.push_back(wanted);
    std::vector<ConstraintProof> premises;
    for (const ClassConstraint &constraint : instance.constraints) {
      std::variant<ConstraintProof, ClassConstraint> premise{
          Entail(SubstituteIn(constraint,
                     [&](unsigned var) {
                       return var < bindings.size() && bindings[var]
                           ? bindings[var]
                           : types_.Named(error_, {});
                     }),
              assumed)};
      if (auto *failed{std::get_if<ClassConstraint>(&premise)}) {
        entailing_.pop_back();
        return *failed;
      }
      premises.push_back(std::get<ConstraintProof>(premise));
    }
    entailing_.pop_back();
    return ConstraintProof{wanted, nullptr, instance.item,
        result_.arena_.SaveArray(llvm::ArrayRef<ConstraintProof>{premises})};
  }
  return wanted;
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
          messages_->Say(Source(unify.tokens),
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
    messages_->Say(Source(unify.tokens),
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
          messages_->Say(Source(unify.tokens),
              "type error in unification of %s and `%s': %s has type %s, but `%s' has type %s"_err_en_US,
              Describe(unify.left), term, Describe(unify.left),
              TypeText(a, VarType(unify.left)), term, TypeText(a, type));
        } else {
          messages_->Say(Source(unify.tokens),
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
  goal_ = unify.number;
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
    failed_ = true;
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
            ? messages_->Say(Source(unify.tokens),
                  "type error in unification of %s and `%s': %s has type %s, which no declaration of %s fits"_err_en_US,
                  Describe(unify.left), term, Describe(unify.left),
                  TypeText(a, VarType(unify.left)), symbol)
            : messages_->Say(Source(unify.tokens),
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
  goal_ = call.number;
  std::vector<Option> options;
  for (const Symbol *candidate : call.name->candidates) {
    options.push_back([&, candidate](Assignment &a) {
      a.symbols[call.number] = candidate;
      const Signature *sig{SignatureOf(*candidate)};
      return !sig || Apply(a, *candidate, *sig, call.arguments, nullptr);
    });
  }
  if (options.empty()) {
    failed_ = true;
    return; // undefined, and reported
  }
  Extend(options, [&](const Assignment &a) {
    names_[call.number] = nullptr;
    std::string types;
    for (VarId argument : call.arguments) {
      types += (types.empty() ? "" : ", ") + TypeText(a, VarType(argument));
    }
    parser::Message &message{messages_->Say(Source(call.tokens),
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
        messages_->Say(Source(call.tokens),
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
                      messages_->Say(Source(t->tokens),
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
                      messages_->Say(Source(t->catchAny->tokens),
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

const Symbol *TypeChecker::Definition(const ClauseForm &clause) const {
  SymbolKind kind{clause.kind == parser::PredOrFunc::Function
          ? SymbolKind::Function
          : SymbolKind::Predicate};
  for (const Symbol *candidate : clause.name->candidates) {
    if (candidate->kind == kind) {
      return candidate;
    }
  }
  return nullptr;
}

ClauseTypes TypeChecker::CheckClause(const ClauseForm &clause, bool inferring) {
  clause_ = &clause;
  tooMany_ = false;
  failed_ = false;
  // What the clause defines, and its types, if any.
  const Symbol *definition{Definition(clause)};
  const Signature *sig{
      definition && !inferring ? SignatureOf(*definition) : nullptr};
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
          messages_->Say(Source(q.tokens),
              "type error: %s has type %s, but it is qualified as %s"_err_en_US,
              Describe(q.var), TypeText(a, VarType(q.var)), TypeText(a, type));
        });
  }
  CheckGoal(clause.body);
  return Finish(clause, definition, sig);
}

ClauseTypes TypeChecker::Finish(const ClauseForm &clause,
    const Symbol *definition, const Signature *sig) {
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
    if (choices.size() < 2 || !names_[goal] || failed_) {
      continue;
    }
    if (!ambiguity) {
      // mmc: "ambiguous overloading causes type ambiguity".
      ambiguity = &messages_->Say(Source(clause.tokens),
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
  bool inferred{sig && definition &&
      std::holds_alternative<std::monostate>(definition->declaration)};
  ClauseTypes types{&clause, definition, sig && !inferred, inferred,
      rigidNames_, {}, {}, {}};
  // The goals' constraints, once the types are known: each is satisfied
  // by the declaration's or by instances (thesis, section 3.1.4). An
  // unsatisfied one is reported once in a clause.
  std::vector<GoalConstraint> proofs;
  if (sig && !ambiguity && !failed_) {
    Assumed assumed{Assume(sig->constraints)};
    llvm::SmallVector<ClassConstraint, 2> reported;
    auto name{[&](unsigned var) {
      return var < rigid_ ? rigidNames_[var].str() : "T_" + std::to_string(var);
    }};
    for (const auto &[goal, instantiated] : chosen.wanted) {
      ClassConstraint wanted{SubstituteIn(instantiated, [&](unsigned var) {
        return Resolve(chosen, types_.Variable(var));
      })};
      bool error{false}, variable{false};
      for (const Type *argument : wanted.arguments) {
        ForEachVariable(*argument, [&](unsigned) { variable = true; });
        error |= HasError(argument);
      }
      // An inferred signature's constraints are not inferred yet.
      if (error || (inferred && variable)) {
        continue;
      }
      std::variant<ConstraintProof, ClassConstraint> entailed{
          Entail(wanted, assumed)};
      if (auto *proof{std::get_if<ConstraintProof>(&entailed)}) {
        proofs.push_back({goal, *proof});
        continue;
      }
      const ClassConstraint &unsatisfied{std::get<ClassConstraint>(entailed)};
      if (llvm::any_of(reported,
              [&](const ClassConstraint &c) { return Same(c, unsatisfied); })) {
        continue;
      }
      reported.push_back(unsatisfied);
      bool undetermined{false};
      for (const Type *argument : unsatisfied.arguments) {
        ForEachVariable(
            *argument, [&](unsigned var) { undetermined |= var >= rigid_; });
      }
      parser::CharBlock where{
          names_[goal] ? Source(names_[goal]->tokens) : Source(clause.tokens)};
      // mmc: "unsatisfiable typeclass constraint" and "unsatisfied
      // typeclass constraint".
      parser::Message &message{undetermined
              ? messages_->Say(where,
                    "unsatisfied type class constraint %s: nothing determines its type"_err_en_US,
                    ConstraintText(unsatisfied, name))
              : messages_->Say(where,
                    "unsatisfiable type class constraint %s"_err_en_US,
                    ConstraintText(unsatisfied, name))};
      if (!Same(unsatisfied, wanted)) {
        message.Attach(where,
            "an instance for %s needs it"_because_en_US,
            ConstraintText(wanted, name));
      }
    }
  }
  types.constraints =
      result_.arena_.SaveArray(llvm::ArrayRef<GoalConstraint>{proofs});
  std::vector<const Type *> vars;
  std::string unbound;
  for (VarId var{0}; var < clause.vars.size(); ++var) {
    const Type *type{Resolve(chosen, VarType(var))};
    vars.push_back(type);
    bool free{false};
    ForEachVariable(*type, [&](unsigned v) { free |= v >= rigid_; });
    // Every variable, as mmc lists them: a term's, such as that of [] in
    // length([]), by the term.
    const VarInfo &info{clause.vars[var]};
    if (free && sig && !ambiguity && !failed_) {
      std::string name{info.kind == VarKind::Program ? info.name.str()
              : info.kind == VarKind::State          ? "!" + info.name.str()
                                                     : Describe(var)};
      if (name != "`'") {
        unbound += "\n  " + name + ": " + TypeText(chosen, type);
      }
    }
  }
  if (!unbound.empty()) {
    // mmc: "warning: unresolved polymorphism".
    messages_->Say(Source(clause.tokens),
        "unresolved polymorphism: the type variables of these types are not determined, and will be bound to void:%s"_warn_en_US,
        unbound);
  }
  types.vars = result_.arena_.SaveArray(llvm::ArrayRef<const Type *>{vars});
  types.symbols =
      result_.arena_.SaveArray(llvm::ArrayRef<const Symbol *>{chosen.symbols});
  types.names =
      result_.arena_.SaveArray(llvm::ArrayRef<const parser::SymName *>{names_});
  return types;
}

//===----------------------------------------------------------------------===//
// Type inference
//===----------------------------------------------------------------------===//

// Calls `f` on each predicate or function a goal may refer to.
void TypeChecker::ForEachReference(
    const Goal &goal, llvm::function_ref<void(const Symbol &)> f) const {
  auto each{[&](llvm::ArrayRef<Goal> goals) {
    for (const Goal &g : goals) {
      ForEachReference(g, f);
    }
  }};
  auto names{[&](const parser::SymName *name) {
    if (name) {
      for (const Symbol *symbol : name->candidates) {
        f(*symbol);
      }
    }
  }};
  common::visit(
      common::visitors{
          [&](const Conj *conj) { each(conj->goals); },
          [&](const Disj *disj) { each(disj->goals); },
          [&](const IfThenElse *ite) {
            ForEachReference(ite->condition, f);
            ForEachReference(ite->then, f);
            ForEachReference(ite->otherwise, f);
          },
          [&](const Not *negation) { ForEachReference(negation->goal, f); },
          [&](const Unify *unify) {
            if (!unify->isVar) {
              names(resolution_.FunctorName(*unify));
              if (unify->functor.lambda) {
                ForEachReference(unify->functor.lambda->body, f);
              }
            }
          },
          [&](const Call *call) {
            if (!call->event) {
              names(call->name);
            }
          },
          [&](const HOCall *) {},
          [&](const Scope *scope) { ForEachReference(scope->goal, f); },
          [&](const Try *t) {
            ForEachReference(t->goal, f);
            ForEachReference(t->then, f);
            if (t->otherwise) {
              ForEachReference(*t->otherwise, f);
            }
            for (const Catch &c : t->catches) {
              ForEachReference(c.pattern, f);
              ForEachReference(c.goal, f);
            }
            if (t->catchAny) {
              ForEachReference(t->catchAny->goal, f);
            }
          },
          [&](const Atomic *atomic) {
            for (const AtomicAlternative &alternative : atomic->alternatives) {
              ForEachReference(alternative.goal, f);
            }
          },
      },
      goal);
}

// The most general signature: a type variable for each argument.
Signature TypeChecker::General(const Symbol &symbol, unsigned arity) {
  llvm::SmallVector<const Type *, 4> arguments;
  for (unsigned j{0}; j < arity; ++j) {
    arguments.push_back(types_.Variable(j));
  }
  Assignment none;
  none.bindings.resize(arity);
  return Generalize(symbol, none, arguments);
}

// A signature of the types, whose variables are numbered from 0 in the
// order they occur, and named T1, T2, ..., as mmc's inferred declarations
// name them.
Signature TypeChecker::Generalize(const Symbol &symbol, const Assignment &a,
    llvm::ArrayRef<const Type *> types) {
  llvm::DenseMap<unsigned, unsigned> numbers;
  llvm::SmallVector<const Type *, 4> arguments;
  for (const Type *type : types) {
    arguments.push_back(types_.Substitute(Resolve(a, type), [&](unsigned var) {
      auto [it, inserted]{numbers.try_emplace(var, numbers.size())};
      return types_.Variable(it->second);
    }));
  }
  std::vector<llvm::StringRef> names;
  for (unsigned j{0}; j < numbers.size(); ++j) {
    names.push_back(result_.arena_.Save("T" + std::to_string(j + 1)));
  }
  return Signature{symbol.kind == SymbolKind::Function
          ? parser::PredOrFunc::Function
          : parser::PredOrFunc::Predicate,
      parser::Purity::Pure, static_cast<unsigned>(numbers.size()),
      result_.arena_.SaveArray(llvm::ArrayRef<llvm::StringRef>{names}),
      result_.arena_.SaveArray(llvm::ArrayRef<const Type *>{arguments})};
}

// The least general type of which both are instances: where they differ, a
// variable for each pair of differing parts.
const Type *TypeChecker::Generalization(const Type *x, const Type *y,
    std::map<std::pair<const Type *, const Type *>, unsigned> &pairs) {
  bool ground{true};
  ForEachVariable(*x, [&](unsigned) { ground = false; });
  if (x == y && ground) {
    return x;
  }
  if (x->kind == TypeKind::Variable || x->kind != y->kind ||
      x->constructor != y->constructor || x->predOrFunc != y->predOrFunc ||
      x->purity != y->purity || x->arguments.size() != y->arguments.size() ||
      !x->result != !y->result) {
    auto [it, inserted]{pairs.try_emplace({x, y}, pairs.size())};
    return types_.Variable(it->second);
  }
  llvm::SmallVector<const Type *, 4> arguments;
  for (auto [a, b] : llvm::zip(x->arguments, y->arguments)) {
    arguments.push_back(Generalization(a, b, pairs));
  }
  const Type *result{x->result ? Generalization(x->result, y->result, pairs)
                               : nullptr};
  switch (x->kind) {
  case TypeKind::Named:
    return types_.Named(*x->constructor, arguments);
  case TypeKind::Tuple:
    return types_.Tuple(arguments);
  default:
    return types_.HigherOrder(x->predOrFunc, x->purity, arguments, result);
  }
}

TypeChecker::Inferred TypeChecker::InferPass(
    const Symbol &symbol, llvm::ArrayRef<const ClauseForm *> clauses) {
  // A clause's assignments may give its head more than one typing, by
  // choosing different declarations for a name. Each combination of one
  // typing of each clause, renamed apart and unified, is a candidate; a
  // clause none of whose typings fits is left out, and its type error is
  // reported when it is checked by the signature.
  struct Candidate {
    Assignment combined;
    std::vector<const Type *> head;
  };
  constexpr std::size_t maxCandidates{64};
  std::vector<Candidate> candidates;
  for (const ClauseForm *clause : clauses) {
    CheckClause(*clause, true);
    // The clause's head typings, each with its variables numbered from 0
    // in the order they occur, so that equal ones are the same.
    std::vector<std::vector<const Type *>> typings;
    unsigned count{0};
    for (const Assignment &a : assignments_) {
      llvm::DenseMap<unsigned, unsigned> numbers;
      std::vector<const Type *> typing;
      for (VarId var : clause->head) {
        typing.push_back(types_.Substitute(
            Resolve(a, VarType(var)), [&](unsigned v) {
              auto [it, inserted]{numbers.try_emplace(v, numbers.size())};
              return types_.Variable(it->second);
            }));
      }
      count = std::max(count, static_cast<unsigned>(numbers.size()));
      if (!llvm::is_contained(typings, typing)) {
        typings.push_back(std::move(typing));
      }
    }
    rigid_ = 0;
    if (typings.empty()) {
      continue;
    }
    if (candidates.empty()) {
      for (std::vector<const Type *> &typing : typings) {
        Assignment combined;
        combined.bindings.resize(count);
        candidates.push_back({std::move(combined), std::move(typing)});
      }
      continue;
    }
    std::vector<Candidate> next;
    for (const Candidate &candidate : candidates) {
      for (const std::vector<const Type *> &typing : typings) {
        if (next.size() == maxCandidates) {
          break;
        }
        Candidate trial{candidate};
        unsigned offset(trial.combined.bindings.size());
        trial.combined.bindings.resize(offset + count);
        bool fits{trial.head.size() == typing.size()};
        for (std::size_t j{0}; fits && j < typing.size(); ++j) {
          const Type *renamed{types_.Substitute(typing[j],
              [&](unsigned v) { return types_.Variable(offset + v); })};
          fits = UnifyTypes(trial.combined, trial.head[j], renamed);
        }
        if (fits) {
          next.push_back(std::move(trial));
        }
      }
    }
    if (!next.empty()) {
      candidates = std::move(next);
    }
  }
  if (candidates.empty()) {
    return {General(symbol, clauses.front()->head.size()), false};
  }
  // The candidates' signatures, without repeats. More than one is an
  // ambiguity, even if one is an instance of another: the clauses then have
  // more than one type assignment (reference manual, "Predicate and
  // function type declarations"), as mmc reports too.
  std::vector<Signature> most;
  for (const Candidate &candidate : candidates) {
    Signature signature{
        Generalize(symbol, candidate.combined, candidate.head)};
    if (!llvm::any_of(most, [&](const Signature &s) {
          return s.arguments == signature.arguments;
        })) {
      most.push_back(signature);
    }
  }
  if (most.size() == 1) {
    return {most.front(), false};
  }
  // Ambiguous: the passes go on with a type of which each is an instance.
  std::vector<const Type *> merged(
      most.front().arguments.begin(), most.front().arguments.end());
  for (std::size_t k{1}; k < most.size(); ++k) {
    std::map<std::pair<const Type *, const Type *>, unsigned> pairs;
    for (std::size_t j{0}; j < merged.size(); ++j) {
      merged[j] = Generalization(merged[j], most[k].arguments[j], pairs);
    }
  }
  unsigned vars{0};
  for (const Type *type : merged) {
    ForEachVariable(*type, [&](unsigned v) { vars = std::max(vars, v + 1); });
  }
  Assignment none;
  none.bindings.resize(vars);
  return {Generalize(symbol, none, merged), true};
}

void TypeChecker::Infer(const Undeclared &undeclared) {
  // The strongly connected components of the undeclared predicates' and
  // functions' call graph, callees first, by Tarjan's algorithm.
  llvm::DenseMap<const Symbol *, unsigned> index, low;
  llvm::DenseSet<const Symbol *> onStack;
  std::vector<const Symbol *> stack;
  std::vector<std::vector<const Symbol *>> components;
  std::function<void(const Symbol &)> visit{[&](const Symbol &symbol) {
    unsigned number(index.size());
    index[&symbol] = low[&symbol] = number;
    stack.push_back(&symbol);
    onStack.insert(&symbol);
    for (const ClauseForm *clause : undeclared.find(&symbol)->second) {
      ForEachReference(clause->body, [&](const Symbol &callee) {
        if (!undeclared.count(&callee)) {
          return;
        }
        if (!index.count(&callee)) {
          visit(callee);
          low[&symbol] = std::min(low[&symbol], low[&callee]);
        } else if (onStack.count(&callee)) {
          low[&symbol] = std::min(low[&symbol], index[&callee]);
        }
      });
    }
    if (low[&symbol] == index[&symbol]) {
      std::vector<const Symbol *> &component{components.emplace_back()};
      const Symbol *member;
      do {
        member = stack.back();
        stack.pop_back();
        onStack.erase(member);
        component.push_back(member);
      } while (member != &symbol);
    }
  }};
  for (const auto &[symbol, clauses] : undeclared) {
    if (!index.count(symbol)) {
      visit(*symbol);
    }
  }
  // Each component's signatures start most general, and each pass checks
  // its clauses by the signatures of the last, until none changes (mmc's
  // --type-inference-iteration-limit bounds the passes too). The passes'
  // messages are dropped: the clauses are checked again, by the final
  // signatures, and report their errors then.
  for (const std::vector<const Symbol *> &component : components) {
    for (const Symbol *symbol : component) {
      const ClauseForm &first{*undeclared.find(symbol)->second.front()};
      unsigned arity(first.head.size());
      result_.signatures_[symbol] = General(*symbol, arity);
    }
    parser::Messages *saved{messages_};
    bool changed{true}, grows{false};
    unsigned pass{0};
    llvm::SmallVector<const Symbol *, 4> ambiguous;
    for (; changed && !grows && pass < options_.inferenceIterationLimit;
         ++pass) {
      changed = false;
      ambiguous.clear();
      for (const Symbol *symbol : component) {
        parser::Messages scratch;
        messages_ = &scratch;
        Inferred inferred{InferPass(*symbol, undeclared.find(symbol)->second)};
        messages_ = saved;
        const Signature &next{inferred.signature};
        if (inferred.ambiguous) {
          ambiguous.push_back(symbol);
        }
        const Signature &last{*result_.signatures_[symbol]};
        if (next.typeVars != last.typeVars ||
            next.arguments != last.arguments) {
          result_.signatures_[symbol] = next;
          changed = true;
          // A type that keeps growing will not settle; this stops it before
          // it is too large to check.
          for (const Type *type : next.arguments) {
            if (TreeSize(*type, maxInferredSize + 1) > maxInferredSize) {
              grows = true;
            }
          }
        }
      }
    }
    if (changed) {
      // mmc: "Type inference iteration limit exceeded". The component is
      // then left without signatures, as if not inferred, so that its last
      // guess draws no more errors.
      const Symbol &symbol{*component.front()};
      parser::CharBlock at{
          Source(undeclared.find(&symbol)->second.front()->tokens)};
      if (grows) {
        messages_->Say(at,
            "type inference did not reach a fixpoint for %s/%u: its type grows in every pass; declare its type"_err_en_US,
            symbol.name.str(), symbol.arity);
      } else {
        messages_->Say(at,
            "type inference did not reach a fixpoint for %s/%u in %u passes; declare its type"_err_en_US,
            symbol.name.str(), symbol.arity, options_.inferenceIterationLimit);
      }
      for (const Symbol *member : component) {
        result_.signatures_[member] = std::nullopt;
      }
    } else {
      // Clauses that fit more than one signature, none more general than
      // the others, are checked without one, and report their ambiguity
      // then, as a declared one's would.
      for (const Symbol *symbol : ambiguous) {
        result_.signatures_[symbol] = std::nullopt;
      }
    }
  }
}

void TypeChecker::Run(const SemanticForm &form) {
  CheckTypeclasses();
  CheckInstances();
  // The module's own declarations, whose errors inference would otherwise
  // report to its scratch list, when it is first to use them.
  for (const Symbol *symbol : resolution_.module().symbols()) {
    if (std::holds_alternative<const parser::EquivalenceTypeDefn *>(
            symbol->declaration)) {
      llvm::SmallVector<const Type *, 4> parameters;
      for (unsigned i{0}; i < symbol->arity; ++i) {
        parameters.push_back(types_.Variable(i));
      }
      Expand(*symbol, parameters, symbol->source);
    } else if (!std::holds_alternative<std::monostate>(symbol->declaration) &&
        symbol->kind != SymbolKind::Type && symbol->kind != SymbolKind::Inst &&
        symbol->kind != SymbolKind::Mode &&
        symbol->kind != SymbolKind::Typeclass) {
      SignatureOf(*symbol);
    }
  }
  // The clauses of each predicate or function without a declaration.
  Undeclared undeclared;
  for (const ClauseForm *clause : form.clauses) {
    const Symbol *definition{Definition(*clause)};
    if (definition &&
        std::holds_alternative<std::monostate>(definition->declaration)) {
      undeclared[definition].push_back(clause);
    }
  }
  if (!options_.inferTypes) {
    // mmc: "clause for predicate `p'/1 without corresponding `:- pred'
    // declaration".
    for (const auto &[symbol, clauses] : undeclared) {
      bool function{symbol->kind == SymbolKind::Function};
      messages_->Say(Source(clauses.front()->tokens),
          "clause for %s %s/%u without corresponding `:- %s' declaration"_err_en_US,
          function ? "function" : "predicate", symbol->name.str(),
          symbol->arity, function ? "func" : "pred");
    }
  }
  Infer(undeclared);
  for (const ClauseForm *clause : form.clauses) {
    result_.clauses_.push_back(CheckClause(*clause, false));
  }
}

std::unique_ptr<TypeChecking> CheckTypes(const SemanticForm &form,
    const NameResolution &resolution, const parser::TokenSequence &tokens,
    parser::Messages &messages, const TypeCheckOptions &options) {
  auto result{std::make_unique<TypeChecking>()};
  TypeChecker{*result, resolution, tokens, messages, options}.Run(form);
  return result;
}

} // namespace merkurius::semantics
