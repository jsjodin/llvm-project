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
#include "llvm/ADT/SmallPtrSet.h"
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
  // With both kinds of constraint, as the reference manual writes them:
  // some [T] (pred p(T, U) => (c(T))) <= (d(U)).
  bool both{!sig.existentialConstraints.empty() && !sig.constraints.empty()};
  if (sig.existentials > 0) {
    os << "some [";
    for (unsigned var{sig.typeVars - sig.existentials}; var < sig.typeVars;
         ++var) {
      os << (var + sig.existentials == sig.typeVars ? "" : ", ") << name(var);
    }
    os << "] ";
  }
  if (both) {
    os << '(';
  }
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
  if (!sig.existentialConstraints.empty()) {
    os << " => (";
    llvm::interleave(
        sig.existentialConstraints, os,
        [&](const ClassConstraint &c) { PrintConstraint(os, c, name); }, ", ");
    os << ')';
  }
  if (both) {
    os << ')';
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
  // The constraints of the goals' callees, instantiated, by goal number;
  // the clause's own number of goals stands for the declaration's
  // existential constraints, which the clause must satisfy.
  std::vector<std::pair<unsigned, ClassConstraint>> wanted;
  // The existential constraints of the goals' callees, on the types that
  // the calls give their existential variables, which the clause assumes.
  std::vector<const ClassConstraint *> given;
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

  // Subtypes (reference manual, "Subtypes"). The supertype of a subtype,
  // with its equivalences expanded, on the subtype's parameters, which are
  // its type variables 0 to n-1; null for another type, or a subtype whose
  // supertype is not a discriminated union type.
  const Type *SupertypeOf(const Symbol &);
  // The variables of a supertype's constructor's types from `first` are
  // its existential variables, which map one to one to those of the
  // subtype's from `subFirst`.
  struct ExistentialMap {
    unsigned first, subFirst;
    llvm::DenseMap<unsigned, unsigned> forward, backward;
  };
  // Whether s =< t, by the subtype definitions; and whether s = t.
  bool IsSubtype(const Type *s, const Type *t, ExistentialMap * = nullptr,
      unsigned depth = 0);
  bool SameType(const Type *s, const Type *t, ExistentialMap *);
  const Symbol *ConstructorOf(
      const Symbol &type, llvm::StringRef name, unsigned arity);
  // The rules for discriminated union types' constructors, and subtypes'.
  void CheckTypeDefinitions();

  // Type conversions (reference manual, "Type conversions"). The base type
  // of a discriminated union type, with its arguments: the type itself, or
  // the last of its chain of supertypes; null for another type, and the
  // error type for a subtype whose supertype is in error.
  const Type *BaseOf(const Type *);
  // For each parameter of a base type, a type of its definition in which
  // the parameter occurs where a coerce must keep it the same; null where
  // it does not.
  llvm::ArrayRef<const Type *> InvariantIn(const Symbol &base);
  void MarkInvariant(const Type *, const Type *context, const Symbol &base,
      std::vector<const Type *> &);
  llvm::DenseMap<const Symbol *, std::vector<const Type *>> invariants_;
  // Why the coerce of the unification cannot convert its argument's type
  // to its result's in the assignment; none if it can, or if either is in
  // error. The types must be known: coerce determines neither.
  std::optional<std::string> CoerceError(const Assignment &, const Unify &);
  // The clause's coerces, checked once its types are known.
  std::vector<const Unify *> coerces_;
  void CheckMethods(const parser::InstanceDecl &, const Symbol &typeclass,
      const std::string &head);
  // The instance's definitions of its type class's methods, with their
  // signatures on its types.
  void DefineMethods(const parser::InstanceDecl &, const Symbol &typeclass,
      llvm::ArrayRef<const Type *> types,
      llvm::ArrayRef<ClassConstraint> constraints,
      llvm::ArrayRef<llvm::StringRef> names, const std::string &text);
  llvm::DenseMap<std::pair<const parser::InstanceDecl *, const Symbol *>,
      const MethodDefinition *>
      methodDefinitions_;
  // Functional dependencies (reference manual, "Functional
  // dependencies"), by the positions of the type class's parameters, each
  // side sorted.
  struct Dependency {
    llvm::SmallVector<unsigned, 2> domain, range;
    bool operator<(const Dependency &that) const {
      return std::tie(domain, range) < std::tie(that.domain, that.range);
    }
    bool operator==(const Dependency &that) const {
      return domain == that.domain && range == that.range;
    }
  };
  std::vector<Dependency> DependenciesOf(const parser::TypeclassDecl &);
  llvm::ArrayRef<Dependency> Dependencies(const Symbol &typeclass);
  std::string DependencyText(const Symbol &typeclass, const Dependency &);
  llvm::DenseMap<const Symbol *, std::vector<Dependency>> dependencies_;
  // Adds to `vars` the type variables that the constraints' functional
  // dependencies, and those of their superclasses, determine from them.
  void Close(llvm::DenseSet<unsigned> &vars,
      llvm::ArrayRef<ClassConstraint> constraints);
  // Improvement (reference manual, "Functional dependencies"): unifies
  // the ranges of the assignment's constraints whose domains are those of
  // another constraint, assumed or wanted, or match an instance's.
  void Improve(Assignment &);
  // The instances that constraints may use: the loaded interfaces' and
  // the module's, without its errors and its abstract declarations.
  using Instance = InstanceHead;

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
  // Whether `wanted` reduces, by the instances, to a ground constraint
  // that neither an instance nor the clause's declaration gives.
  bool Refuted(const ClassConstraint &wanted, const Assumed &);
  bool Refuted(const ClassConstraint &wanted) {
    return Refuted(wanted, assumed_);
  }
  bool Refuted(const Assignment &);
  Assumed assumed_; // the clause's declaration's
  // The declaration's and those the assignment's calls give.
  Assumed AssumedBy(const Assignment &);
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
  // A clause of an instance's method is checked by its definition's
  // signature.
  ClauseTypes CheckClause(const ClauseForm &, bool inferring,
      const MethodDefinition *method = nullptr);
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
  // `given` are constraints the calls give on existential types.
  Signature Generalize(const Symbol &, const Assignment &,
      llvm::ArrayRef<const Type *>, llvm::ArrayRef<ClassConstraint> = {},
      llvm::ArrayRef<ClassConstraint> given = {});
  // The constraints that an assignment's goals need, reduced by the
  // instances, that a signature must assume: those on the head's type
  // variables, which `numbers` renumbers.
  std::vector<ClassConstraint> Residual(const Assignment &,
      llvm::DenseMap<unsigned, unsigned> &numbers);
  bool Reduce(const ClassConstraint &, std::vector<ClassConstraint> &);
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
  const Type *VarType(VarId var) { return types_.Variable(base_ + var); }
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
  // The declaration's universally quantified type variables, which the
  // clause cannot bind, and then its existentially quantified ones, which
  // the clause determines; the clause's variables' types are numbered from
  // base_.
  unsigned rigid_{0};
  unsigned base_{0};
  // The types that the calls of the clause give their callees' existential
  // variables, by goal, callee and variable, and their existential
  // constraints, by goal, callee and number, the same in each assignment.
  std::map<std::tuple<unsigned, const Symbol *, unsigned>, const Type *>
      callExistentials_;
  std::map<std::tuple<unsigned, const Symbol *, unsigned>,
      const ClassConstraint *>
      callGiven_;
  llvm::DenseMap<const ClassConstraint *, unsigned> givenGoals_;
  unsigned nextExistential_{0};
  // The types an existentially typed predicate's or function's clauses,
  // or an instance's method's, have given its existential variables, by
  // the definition or the MethodDefinition: all its clauses give the same
  // (reference manual, "Semantics of type quantifiers").
  llvm::DenseMap<const void *, std::vector<const Type *>> existentialTypes_;
  // The field updates that cannot be made, with the existential variable
  // that the field shares with another.
  struct NotUpdatable {
    llvm::StringRef var;
    parser::CharBlock field; // its definition
  };
  llvm::DenseMap<const Symbol *, NotUpdatable> notUpdatable_;
  // The signature of 'new f', a construction with an existentially typed
  // constructor f: the caller chooses the existential variables, and must
  // satisfy their constraints (reference manual, "Existentially typed
  // data types").
  const Signature &NewSignature(const Symbol &, const Signature &);
  std::map<const Symbol *, Signature> newSignatures_;
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
    // functions"); a method's, its type class's parameters too, unless
    // functional dependencies determine it from them.
    llvm::DenseSet<unsigned> determined;
    for (const Type *argument : arguments) {
      ForEachVariable(*argument, [&](unsigned var) { determined.insert(var); });
    }
    // and those that functional dependencies determine from them.
    Close(determined, constraints);
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
    // The existentially quantified type variables and their constraints
    // (reference manual, "Existential class constraints"): an existential
    // constraint constrains only existentially quantified variables, and a
    // universal one only universally quantified ones.
    llvm::SmallVector<ClassConstraint, 2> existentialConstraints;
    for (const parser::ClassConstraint *c : (*decl)->existentialConstraints) {
      if (std::optional<ClassConstraint> converted{
              ConvertConstraint(*c, names)}) {
        existentialConstraints.push_back(*converted);
      }
    }
    llvm::DenseSet<unsigned> existential;
    for (llvm::StringRef var : (*decl)->existentialVars) {
      if (std::optional<unsigned> number{names.Get(var)};
          number && !existential.insert(*number).second) {
        // mmc crashes on this (see the register).
        messages_->Say(Source((*decl)->tokens),
            "type variable %s is quantified twice in the declaration of %s %s/%u"_err_en_US,
            var.str(), function ? "function" : "predicate", symbol.name.str(),
            symbol.arity);
      }
    }
    // A constraint that breaks the rule is left out, once reported.
    llvm::DenseSet<unsigned> misquantified;
    for (bool existentially : {true, false}) {
      auto &list{existentially ? existentialConstraints : constraints};
      llvm::erase_if(list, [&](const ClassConstraint &constraint) {
        bool wrong{false};
        for (const Type *argument : constraint.arguments) {
          ForEachVariable(*argument, [&](unsigned var) {
            if (existential.contains(var) == existentially) {
              return;
            }
            wrong = true;
            if (misquantified.insert(var).second) {
              // mmc: "type variable `U' is existentially constrained, but
              // is universally quantified".
              messages_->Say(Source((*decl)->tokens),
                  "type variable %s of %s %s/%u is %s constrained, but %s quantified"_err_en_US,
                  names.names[var].str(), function ? "function" : "predicate",
                  symbol.name.str(), symbol.arity,
                  existentially ? "existentially" : "universally",
                  existentially ? "universally" : "existentially");
            }
          });
        }
        return wrong;
      });
    }
    // The existential variables are numbered last.
    if (!existential.empty()) {
      std::vector<unsigned> order(names.next);
      unsigned number{0};
      for (bool last : {false, true}) {
        for (unsigned var{0}; var < names.next; ++var) {
          if (existential.contains(var) == last) {
            order[var] = number++;
          }
        }
      }
      auto renumber{[&](unsigned var) { return types_.Variable(order[var]); }};
      for (const Type *&argument : arguments) {
        argument = types_.Substitute(argument, renumber);
      }
      for (ClassConstraint &constraint : constraints) {
        constraint = SubstituteIn(constraint, renumber);
      }
      for (ClassConstraint &constraint : existentialConstraints) {
        constraint = SubstituteIn(constraint, renumber);
      }
      std::vector<llvm::StringRef> renamed(names.next);
      for (unsigned var{0}; var < names.next && var < names.names.size();
           ++var) {
        renamed[order[var]] = names.names[var];
      }
      names.names = std::move(renamed);
    }
    Signature sig{make((*decl)->kind, (*decl)->purity)};
    sig.existentials = existential.size();
    sig.existentialConstraints = result_.arena_.SaveArray(
        llvm::ArrayRef<ClassConstraint>{existentialConstraints});
    return sig;
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
    // The existentially quantified variables are numbered last
    // (reference manual, "Existentially typed data types"). The
    // definition's rules are checked with the constructor's own symbol.
    bool report{symbol.kind == SymbolKind::Constructor};
    auto constructorName{[&] {
      return symbol.name.str() + "/" +
          std::to_string((*constructor)->arguments.size());
    }};
    for (llvm::StringRef var : (*constructor)->existentialVars) {
      if (names.numbers.count(var)) {
        // mmc: "type variable `T' has overlapping scopes".
        if (report) {
          messages_->Say(Source((*constructor)->tokens),
              "type variable %s of constructor %s is a parameter of its type, and cannot be existentially quantified too"_err_en_US,
              var.str(), constructorName());
        }
        continue;
      }
      names.Add(var);
    }
    unsigned universal(parameters.size());
    names.closed = true;
    llvm::SmallVector<const Type *, 4> fields;
    for (const parser::ConstructorArg *argument : (*constructor)->arguments) {
      fields.push_back(Convert(argument->type, names));
    }
    llvm::SmallVector<ClassConstraint, 2> existentialConstraints;
    for (const parser::ClassConstraint *c : (*constructor)->constraints) {
      if (std::optional<ClassConstraint> converted{
              ConvertConstraint(*c, names)}) {
        existentialConstraints.push_back(*converted);
      }
    }
    if (report) {
      llvm::DenseSet<unsigned> inArguments, inConstraints, reported;
      for (const Type *field : fields) {
        ForEachVariable(*field, [&](unsigned var) { inArguments.insert(var); });
      }
      llvm::DenseSet<unsigned> determined{inArguments};
      Close(determined, existentialConstraints);
      for (const ClassConstraint &constraint : existentialConstraints) {
        for (const Type *argument : constraint.arguments) {
          ForEachVariable(*argument, [&](unsigned var) {
            inConstraints.insert(var);
            if (!reported.insert(var).second) {
              return;
            }
            if (var < universal) {
              // mmc: "the type variable `U' occurs in a class constraint
              // without being explicitly existentially quantified".
              messages_->Say(Source((*constructor)->tokens),
                  "type variable %s is in a constraint of constructor %s, but is not existentially quantified"_err_en_US,
                  names.names[var].str(), constructorName());
            } else if (!determined.contains(var)) {
              // mmc: "type variable `T2' occurs in the constraints, but is
              // not determined by the constructor's argument types".
              messages_->Say(Source((*constructor)->tokens),
                  "type variable %s occurs in the constraints of constructor %s, but its argument types do not determine it"_err_en_US,
                  names.names[var].str(), constructorName());
            }
          });
        }
      }
      for (unsigned var{universal}; var < names.next; ++var) {
        if (!inArguments.contains(var) && !inConstraints.contains(var)) {
          // mmc: "the existentially quantified type variable `T' does not
          // occur either in the arguments or in the constraints".
          messages_->Say(Source((*constructor)->tokens),
              "existentially quantified type variable %s of constructor %s occurs neither in its arguments nor in its constraints"_err_en_US,
              names.names[var].str(), constructorName());
        }
      }
    }
    // A constraint on a type parameter is reported, and left out.
    llvm::erase_if(existentialConstraints, [&](const ClassConstraint &c) {
      bool parameter{false};
      for (const Type *argument : c.arguments) {
        ForEachVariable(
            *argument, [&](unsigned var) { parameter |= var < universal; });
      }
      return parameter;
    });
    const Type *owner{types_.Named(*symbol.owner, parameters)};
    // A constructor, as a deconstruction, and a field's selection give the
    // existential variables, as an existentially typed function does; the
    // construction 'new f' is the universal version (see NewSignature).
    unsigned existentials{names.next - universal};
    if (symbol.kind == SymbolKind::Constructor) {
      arguments.append(fields.begin(), fields.end());
      arguments.push_back(owner);
    } else {
      // f/1 is (T) = F, and 'f :='/2 is (T, F) = T.
      llvm::StringRef field{symbol.name};
      field.consume_back(" :=");
      const Type *type{types_.Named(error_, {})};
      for (auto [argument, converted] :
          llvm::zip((*constructor)->arguments, fields)) {
        if (argument->field && argument->field->name == field) {
          type = converted;
        }
      }
      arguments.push_back(owner);
      arguments.push_back(type);
      if (symbol.kind == SymbolKind::FieldUpdate) {
        arguments.push_back(owner);
        // An update constructs: it chooses the field's existential
        // variables, whose constraints it must satisfy, and keeps the
        // others, which another field must not share (mmc: "Field `a'
        // cannot be updated because the existentially quantified type
        // variable `T' occurs in the types of field `a' and some other
        // field").
        llvm::DenseSet<unsigned> own;
        ForEachVariable(*type, [&](unsigned var) {
          if (var >= universal) {
            own.insert(var);
          }
        });
        for (auto [argument, converted] :
            llvm::zip((*constructor)->arguments, fields)) {
          if (argument->field && argument->field->name == field) {
            continue;
          }
          ForEachVariable(*converted, [&](unsigned var) {
            if (own.contains(var)) {
              const parser::ConstructorArg *updated{nullptr};
              for (const parser::ConstructorArg *a : (*constructor)->arguments) {
                if (a->field && a->field->name == field) {
                  updated = a;
                }
              }
              notUpdatable_[&symbol] = {names.names[var],
                  Source(updated ? updated->tokens : (*constructor)->tokens)};
            }
          });
        }
        for (const ClassConstraint &c : existentialConstraints) {
          bool mentions{false};
          for (const Type *argument : c.arguments) {
            ForEachVariable(*argument,
                [&](unsigned var) { mentions |= own.contains(var); });
          }
          if (mentions) {
            constraints.push_back(c);
          }
        }
        existentialConstraints.clear();
        existentials = 0;
      }
    }
    Signature sig{make(parser::PredOrFunc::Function, parser::Purity::Pure)};
    sig.existentials = existentials;
    sig.existentialConstraints = result_.arena_.SaveArray(
        llvm::ArrayRef<ClassConstraint>{existentialConstraints});
    return sig;
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
  if (x->kind != y->kind || x->var != y->var || x->constructor != y->constructor ||
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
  // The callee determines its existential variables, each a type the
  // caller cannot bind, and assures their constraints (reference manual,
  // "Semantics of type quantifiers" and "Existential class constraints").
  for (unsigned var{sig.typeVars - sig.existentials}; var < sig.typeVars;
       ++var) {
    const Type *&existential{callExistentials_[{goal_, &symbol, var}]};
    if (!existential) {
      existential = types_.Existential(nextExistential_++,
          var < sig.varNames.size() ? sig.varNames[var] : "T");
    }
    a.bindings[offset + var] = existential;
  }
  for (unsigned j{0}; j < sig.existentialConstraints.size(); ++j) {
    const ClassConstraint *&given{callGiven_[{goal_, &symbol, j}]};
    if (!given) {
      given = result_.arena_.Make<ClassConstraint>(SubstituteIn(
          sig.existentialConstraints[j], [&](unsigned var) {
            return Resolve(a, types_.Variable(offset + var));
          }));
      givenGoals_[given] = goal_;
    }
    a.given.push_back(given);
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
  // higher-order terms"). Of an existentially typed predicate or function
  // there is none, as in mmc: only a call determines its existential
  // types, which the closure's type would need (see the register).
  if (sig.existentials > 0) {
    return false;
  }
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
  // Assignments whose constraints cannot be satisfied are dropped, if
  // others are left, as mmc does: so show(2.0 + 1.0), with an instance
  // only for show(int), takes int's +. If none is left, the clause
  // reports the constraint.
  if (extended.size() > 1) {
    std::vector<Assignment> satisfiable;
    for (Assignment &a : extended) {
      if (!Refuted(a)) {
        satisfiable.push_back(std::move(a));
      }
    }
    if (satisfiable.empty()) {
      // Moved from: check them again.
      for (const Assignment &a : assignments_) {
        for (const Option &option : options) {
          Assignment copy{a};
          if (option(copy)) {
            satisfiable.push_back(std::move(copy));
          }
        }
      }
    }
    extended = std::move(satisfiable);
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

std::vector<TypeChecker::Dependency> TypeChecker::DependenciesOf(
    const parser::TypeclassDecl &decl) {
  std::vector<Dependency> dependencies;
  auto positions{[&](llvm::ArrayRef<llvm::StringRef> vars) {
    llvm::SmallVector<unsigned, 2> result;
    for (llvm::StringRef var : vars) {
      // The parser checks that each is a parameter.
      auto it{llvm::find(decl.parameters, var)};
      if (it != decl.parameters.end()) {
        result.push_back(it - decl.parameters.begin());
      }
    }
    llvm::sort(result);
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }};
  for (const parser::TypeclassConstraint &item : decl.constraints) {
    if (const auto *dependency{
            std::get_if<const parser::FunctionalDependency *>(&item)}) {
      dependencies.push_back(Dependency{
          positions((*dependency)->domain), positions((*dependency)->range)});
    }
  }
  llvm::sort(dependencies);
  dependencies.erase(std::unique(dependencies.begin(), dependencies.end()),
      dependencies.end());
  return dependencies;
}

llvm::ArrayRef<TypeChecker::Dependency> TypeChecker::Dependencies(
    const Symbol &typeclass) {
  auto [it, inserted]{dependencies_.try_emplace(&typeclass)};
  if (inserted) {
    if (const auto *decl{std::get_if<const parser::TypeclassDecl *>(
            &typeclass.declaration)}) {
      it->second = DependenciesOf(**decl);
    }
  }
  return it->second;
}

std::string TypeChecker::DependencyText(
    const Symbol &typeclass, const Dependency &dependency) {
  const auto *decl{
      std::get_if<const parser::TypeclassDecl *>(&typeclass.declaration)};
  auto side{[&](llvm::ArrayRef<unsigned> positions) {
    std::string text;
    for (unsigned position : positions) {
      if (!text.empty()) {
        text += ", ";
      }
      text += decl && position < (*decl)->parameters.size()
          ? (*decl)->parameters[position].str()
          : "T" + std::to_string(position);
    }
    return text;
  }};
  return "(" + side(dependency.domain) + " -> " + side(dependency.range) + ")";
}

void TypeChecker::Close(llvm::DenseSet<unsigned> &vars,
    llvm::ArrayRef<ClassConstraint> constraints) {
  // The induced dependencies: of each constraint, and of its superclass
  // constraints on its arguments, recursively.
  std::vector<std::pair<llvm::DenseSet<unsigned>, llvm::DenseSet<unsigned>>>
      induced;
  std::function<void(const ClassConstraint &, unsigned)> induce{
      [&](const ClassConstraint &constraint, unsigned depth) {
        auto varsAt{[&](llvm::ArrayRef<unsigned> positions) {
          llvm::DenseSet<unsigned> result;
          for (unsigned position : positions) {
            if (position < constraint.arguments.size()) {
              ForEachVariable(*constraint.arguments[position],
                  [&](unsigned var) { result.insert(var); });
            }
          }
          return result;
        }};
        for (const Dependency &dependency :
            Dependencies(*constraint.typeclass)) {
          induced.emplace_back(
              varsAt(dependency.domain), varsAt(dependency.range));
        }
        if (depth > 64) {
          return; // a superclass cycle, which is reported
        }
        for (const ClassConstraint &super :
            Superclasses(*constraint.typeclass)) {
          induce(SubstituteIn(super,
                     [&](unsigned var) {
                       return var < constraint.arguments.size()
                           ? constraint.arguments[var]
                           : types_.Named(error_, {});
                     }),
              depth + 1);
        }
      }};
  for (const ClassConstraint &constraint : constraints) {
    induce(constraint, 0);
  }
  for (bool changed{true}; changed;) {
    changed = false;
    for (const auto &[domain, range] : induced) {
      if (llvm::all_of(domain, [&](unsigned var) { return vars.contains(var); })) {
        for (unsigned var : range) {
          changed |= vars.insert(var).second;
        }
      }
    }
  }
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
  if (auto it{result_.superclasses_.find(&typeclass)};
      it != result_.superclasses_.end()) {
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
  result_.superclasses_[&typeclass] = saved;
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
      // mmc: "The functional dependencies do not match."
      if (llvm::ArrayRef<Dependency>{DependenciesOf(*abstract->decl)} !=
          Dependencies(*symbol)) {
        parser::Message &message{messages_->Say(where(*definition->decl),
            "the functional dependencies of typeclass %s/%u differ from those of its abstract declaration"_err_en_US,
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
  // Functional dependencies: in each of the module's instances, the range's
  // type variables are determined by the domain's, with the instance's
  // constraints; and two instances whose domains unify have the same
  // range, with the unifier (reference manual, "Functional dependencies").
  unsigned savedRigid{rigid_};
  rigid_ = 0;
  auto countVars{[](llvm::ArrayRef<const Type *> types) {
    unsigned count{0};
    for (const Type *type : types) {
      ForEachVariable(
          *type, [&](unsigned var) { count = std::max(count, var + 1); });
    }
    return count;
  }};
  for (std::size_t j{0}; j < heads.size(); ++j) {
    const Head &head{heads[j]};
    if (!head.item->local || !head.valid || head.item->decl->abstract) {
      continue;
    }
    InTokensOf in{*this, head.item->tokens};
    auto varName{[&](unsigned var) {
      return var < head.names.size() ? head.names[var].str()
                                     : "T" + std::to_string(var);
    }};
    for (const Dependency &dependency : Dependencies(*head.typeclass)) {
      llvm::DenseSet<unsigned> determined;
      for (unsigned position : dependency.domain) {
        ForEachVariable(*head.types[position],
            [&](unsigned var) { determined.insert(var); });
      }
      Close(determined, head.constraints);
      llvm::DenseSet<unsigned> reported;
      for (unsigned position : dependency.range) {
        ForEachVariable(*head.types[position], [&](unsigned var) {
          if (!determined.contains(var) && reported.insert(var).second) {
            // mmc: "functional dependency not satisfied: type variable `T'
            // occurs in the range of the functional dependency, but is not
            // determined by the domain."
            messages_->Say(Source(head.item->decl->name->tokens),
                "in instance %s, type variable %s is in the range of the functional dependency %s, but its domain does not determine it"_err_en_US,
                head.text, varName(var),
                DependencyText(*head.typeclass, dependency));
          }
        });
      }
    }
    for (std::size_t k{0}; k < j; ++k) {
      const Head &other{heads[k]};
      if (!other.valid || other.typeclass != head.typeclass ||
          (other.item->local && other.item->decl->abstract) ||
          other.key == head.key) {
        continue; // an overlap is reported as one
      }
      unsigned otherVars{countVars(other.types)};
      unsigned headVars{countVars(head.types)};
      auto renamed{[&](const Type *type) {
        return types_.Substitute(
            type, [&](unsigned var) { return types_.Variable(otherVars + var); });
      }};
      for (const Dependency &dependency : Dependencies(*head.typeclass)) {
        Assignment a;
        a.bindings.resize(otherVars + headVars);
        bool unify{llvm::all_of(dependency.domain, [&](unsigned position) {
          return UnifyTypes(
              a, other.types[position], renamed(head.types[position]));
        })};
        if (!unify ||
            llvm::all_of(dependency.range, [&](unsigned position) {
              return Resolve(a, other.types[position]) ==
                  Resolve(a, renamed(head.types[position]));
            })) {
          continue;
        }
        // mmc: "Inconsistent instance declaration for typeclass `m.baz'/2
        // with functional dependency `(A -> B)'."
        parser::Message &message{messages_->Say(
            Source(head.item->decl->name->tokens),
            "instance %s is inconsistent with an earlier instance by the functional dependency %s of typeclass %s/%u"_err_en_US,
            head.text, DependencyText(*head.typeclass, dependency),
            head.typeclass->name.str(), head.typeclass->arity)};
        InTokensOf otherIn{*this, other.item->tokens};
        message.Attach(Source(other.item->decl->name->tokens),
            "the earlier instance is %s"_because_en_US, other.text);
        break;
      }
    }
  }
  rigid_ = savedRigid;
  for (const Head &head : heads) {
    if (head.valid && !(head.item->local && head.item->decl->abstract)) {
      result_.instances_.push_back({head.item, head.typeclass,
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
      DefineMethods(decl, *head.typeclass, head.types, head.constraints,
          head.names, head.text);
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

const Type *TypeChecker::SupertypeOf(const Symbol &symbol) {
  if (auto it{result_.supertypes_.find(&symbol)}; it != result_.supertypes_.end()) {
    return it->second;
  }
  result_.supertypes_[&symbol] = nullptr;
  const auto *defn{
      std::get_if<const parser::DuTypeDefn *>(&symbol.declaration)};
  if (!defn || !(*defn)->supertype) {
    return nullptr;
  }
  InTokensOf in{*this, symbol};
  TypeVarNames names;
  for (llvm::StringRef parameter : (*defn)->parameters) {
    names.Add(parameter);
  }
  names.closed = true;
  const Type *super{Convert(*(*defn)->supertype, names)};
  if (HasError(super)) {
    return nullptr; // reported
  }
  if (super->kind != TypeKind::Named ||
      !std::holds_alternative<const parser::DuTypeDefn *>(
          super->constructor->declaration)) {
    // mmc: "`int'/0, the declared super type of `bad5'/0, is not a
    // discriminated union type".
    messages_->Say(Source((*defn)->tokens),
        "the supertype %s of %s/%u is not a discriminated union type whose definition is visible"_err_en_US,
        TypeToString(*super, [&](unsigned var) {
          return names.names[var].str();
        }),
        symbol.name.str(), symbol.arity);
    return nullptr;
  }
  result_.supertypes_[&symbol] = super;
  return super;
}

bool TypeChecker::IsSubtype(
    const Type *s, const Type *t, ExistentialMap *map, unsigned depth) {
  if (s == t) {
    return true;
  }
  if (depth > 64 || s->kind != TypeKind::Named || t->kind != TypeKind::Named) {
    // Higher-order types and others are subtypes only of themselves.
    return SameType(s, t, map);
  }
  if (s->constructor == t->constructor) {
    // The same type constructor: argument by argument, for a
    // discriminated union type.
    bool du{std::holds_alternative<const parser::DuTypeDefn *>(
        s->constructor->declaration)};
    for (auto [x, y] : llvm::zip(s->arguments, t->arguments)) {
      if (!(du ? IsSubtype(x, y, map, depth + 1) : SameType(x, y, map))) {
        return false;
      }
    }
    return true;
  }
  // s's supertype, with its arguments, must be a subtype of t.
  const Type *super{SupertypeOf(*s->constructor)};
  if (!super) {
    return false;
  }
  return IsSubtype(types_.Substitute(super,
                       [&](unsigned var) {
                         return var < s->arguments.size()
                             ? s->arguments[var]
                             : types_.Named(error_, {});
                       }),
      t, map, depth + 1);
}

bool TypeChecker::SameType(const Type *s, const Type *t, ExistentialMap *map) {
  if (map && t->kind == TypeKind::Variable && t->var >= map->first) {
    if (s->kind != TypeKind::Variable || s->var < map->subFirst) {
      return false;
    }
    auto forward{map->forward.try_emplace(t->var, s->var).first};
    auto backward{map->backward.try_emplace(s->var, t->var).first};
    return forward->second == s->var && backward->second == t->var;
  }
  if (s == t || IsError(s) || IsError(t)) {
    return true;
  }
  if (s->kind != t->kind || s->kind == TypeKind::Variable ||
      s->kind == TypeKind::Existential || s->constructor != t->constructor ||
      s->predOrFunc != t->predOrFunc || s->purity != t->purity ||
      s->arguments.size() != t->arguments.size() ||
      (s->result == nullptr) != (t->result == nullptr)) {
    return false;
  }
  for (auto [x, y] : llvm::zip(s->arguments, t->arguments)) {
    if (!SameType(x, y, map)) {
      return false;
    }
  }
  return !s->result || SameType(s->result, t->result, map);
}

const Symbol *TypeChecker::ConstructorOf(
    const Symbol &type, llvm::StringRef name, unsigned arity) {
  const ModuleSymbols *symbols{type.module == resolution_.module().name()
          ? &resolution_.module()
          : resolution_.Module(type.module)};
  if (!symbols) {
    return nullptr;
  }
  for (const Symbol *symbol : symbols->Lookup(name)) {
    if (symbol->kind == SymbolKind::Constructor && symbol->owner == &type &&
        symbol->arity == arity) {
      return symbol;
    }
  }
  return nullptr;
}

void TypeChecker::CheckTypeDefinitions() {
  for (const Symbol *symbol : resolution_.module().symbols()) {
    const auto *defn{
        std::get_if<const parser::DuTypeDefn *>(&symbol->declaration)};
    if (symbol->kind != SymbolKind::Type || !defn) {
      continue;
    }
    InTokensOf in{*this, *symbol};
    // mmc: "constructor `m.lemon/0' for type `m.c7/0' multiply defined".
    llvm::StringSet<> seen;
    for (const parser::Constructor *constructor : (*defn)->constructors) {
      std::string key{constructor->name->name.str() + "/" +
          std::to_string(constructor->arguments.size())};
      if (!seen.insert(key).second) {
        messages_->Say(Source(constructor->tokens),
            "constructor %s is defined twice in type %s/%u"_err_en_US, key,
            symbol->name.str(), symbol->arity);
      }
    }
    if (!(*defn)->supertype) {
      continue;
    }
    const Type *super{SupertypeOf(*symbol)};
    if (!super) {
      continue;
    }
    // The chain of supertypes ends at a base type, which is not a subtype.
    llvm::SmallVector<const Symbol *, 4> chain{symbol};
    bool circular{false};
    for (const Type *t{super}; t;) {
      const Symbol *next{t->constructor};
      if (next == symbol) {
        circular = true;
        break;
      }
      if (llvm::is_contained(chain, next)) {
        break; // a circularity reported for its own types
      }
      chain.push_back(next);
      t = SupertypeOf(*next);
    }
    if (circular) {
      // mmc: "circularity in subtype definition chain".
      parser::Message &message{messages_->Say(Source((*defn)->tokens),
          "the chain of supertypes of %s/%u is circular"_err_en_US,
          symbol->name.str(), symbol->arity)};
      for (const Symbol *link : llvm::ArrayRef(chain).drop_front()) {
        message.Attach(link->source, "it goes through %s.%s/%u"_because_en_US,
            link->module.str(), link->name.str(), link->arity);
      }
      continue;
    }
    if ((*defn)->attributes &&
        ((*defn)->attributes->equality || (*defn)->attributes->comparison)) {
      // mmc: "the subtype `c6'/0 is not allowed to have its own
      // user-defined equality or comparison".
      messages_->Say(Source((*defn)->tokens),
          "the subtype %s/%u cannot have its own equality or comparison; it has its base type's"_err_en_US,
          symbol->name.str(), symbol->arity);
    }
    const Symbol &superSymbol{*super->constructor};
    std::string superText{TypeToString(*super, [&](unsigned var) {
      return var < (*defn)->parameters.size()
          ? (*defn)->parameters[var].str()
          : "T" + std::to_string(var);
    })};
    const auto *superDefn{
        std::get<const parser::DuTypeDefn *>(superSymbol.declaration)};
    unsigned parameters((*defn)->parameters.size());
    long last{-1};
    bool ordered{true};
    for (const parser::Constructor *constructor : (*defn)->constructors) {
      llvm::StringRef name{constructor->name->name};
      unsigned arity(constructor->arguments.size());
      // A subtype's constructors are among its supertype's.
      const Symbol *superConstructor{ConstructorOf(superSymbol, name, arity)};
      const Symbol *own{ConstructorOf(*symbol, name, arity)};
      if (!superConstructor) {
        // mmc: "`banana'/0 is not a constructor of the supertype
        // `fruit'/0".
        messages_->Say(Source(constructor->tokens),
            "%s/%u is not a constructor of the supertype %s"_err_en_US,
            name.str(), arity, superText);
        continue;
      }
      for (std::size_t j{0}; j < superDefn->constructors.size(); ++j) {
        const parser::Constructor *c{superDefn->constructors[j]};
        if (c->name->name == name && c->arguments.size() == arity) {
          ordered &= static_cast<long>(j) > last;
          last = j;
        }
      }
      const Signature *ownSig{own ? SignatureOf(*own) : nullptr};
      const Signature *superSig{SignatureOf(*superConstructor)};
      if (!ownSig || !superSig) {
        continue;
      }
      // Each argument's type is a subtype of the supertype's, whose
      // parameters are the supertype's arguments, and whose existential
      // variables correspond one to one to the subtype's.
      unsigned superParameters{superSig->typeVars - superSig->existentials};
      ExistentialMap map{ownSig->typeVars, parameters, {}, {}};
      auto instantiate{[&](unsigned var) {
        return var < superParameters
            ? (var < super->arguments.size() ? super->arguments[var]
                                             : types_.Named(error_, {}))
            : types_.Variable(map.first + var - superParameters);
      }};
      auto ownName{[&](unsigned var) {
        return var < ownSig->varNames.size() && !ownSig->varNames[var].empty()
            ? ownSig->varNames[var].str()
            : "T" + std::to_string(var);
      }};
      auto superName{[&](unsigned var) {
        unsigned original{var - map.first + superParameters};
        return var >= map.first && original < superSig->varNames.size()
            ? superSig->varNames[original].str()
            : ownName(var);
      }};
      for (unsigned j{0}; j < arity; ++j) {
        const Type *t{types_.Substitute(superSig->arguments[j], instantiate)};
        if (!IsSubtype(ownSig->arguments[j], t, &map)) {
          // mmc: "the first argument of `basket' has a type, `int', which
          // is not a subtype of the corresponding argument type
          // `m.fruit' in the supertype".
          messages_->Say(Source(constructor->tokens),
              "argument %u of %s/%u has type %s, which is not a subtype of %s, its type in the supertype"_err_en_US,
              j + 1, name.str(), arity,
              TypeToString(*ownSig->arguments[j], ownName),
              TypeToString(*t, superName));
        }
      }
      // The same existential variables and constraints.
      bool same{ownSig->existentials == superSig->existentials &&
          map.forward.size() == superSig->existentials &&
          ownSig->existentialConstraints.size() ==
              superSig->existentialConstraints.size()};
      for (const ClassConstraint &c : superSig->existentialConstraints) {
        if (!same) {
          break;
        }
        ClassConstraint mapped{SubstituteIn(c, [&](unsigned var) {
          const Type *t{instantiate(var)};
          if (t->kind == TypeKind::Variable && t->var >= map.first) {
            auto it{map.forward.find(t->var)};
            return it == map.forward.end() ? types_.Named(error_, {})
                                           : types_.Variable(it->second);
          }
          return t;
        })};
        same &= llvm::any_of(ownSig->existentialConstraints,
            [&](const ClassConstraint &x) { return Same(x, mapped); });
      }
      if (!same && (ownSig->existentials > 0 || superSig->existentials > 0)) {
        messages_->Say(Source(constructor->tokens),
            "%s/%u must have the existentially quantified type variables and constraints of the supertype's constructor, one to one"_err_en_US,
            name.str(), arity);
      }
    }
    if (!ordered) {
      // mmc: "`c4'/0 declares some constructors in a different order to
      // its supertype `fruit'/0".
      messages_->Say(Source((*defn)->tokens),
          "the constructors of %s/%u are in a different order from those of its supertype %s"_warn_en_US,
          symbol->name.str(), symbol->arity, superText);
    }
  }
}

const Type *TypeChecker::BaseOf(const Type *type) {
  llvm::SmallPtrSet<const Symbol *, 4> seen;
  while (type->kind == TypeKind::Named &&
      std::holds_alternative<const parser::DuTypeDefn *>(
          type->constructor->declaration)) {
    if (!std::get<const parser::DuTypeDefn *>(type->constructor->declaration)
             ->supertype) {
      return type;
    }
    const Type *super{SupertypeOf(*type->constructor)};
    if (!super || !seen.insert(type->constructor).second) {
      return types_.Named(error_, {}); // reported
    }
    llvm::ArrayRef<const Type *> arguments{type->arguments};
    type = types_.Substitute(super, [&](unsigned var) {
      return var < arguments.size() ? arguments[var] : types_.Named(error_, {});
    });
  }
  return nullptr;
}

llvm::ArrayRef<const Type *> TypeChecker::InvariantIn(const Symbol &base) {
  if (auto it{invariants_.find(&base)}; it != invariants_.end()) {
    return it->second;
  }
  std::vector<const Type *> invariant(base.arity, nullptr);
  const auto *defn{std::get<const parser::DuTypeDefn *>(base.declaration)};
  for (const parser::Constructor *constructor : defn->constructors) {
    const Symbol *symbol{ConstructorOf(
        base, constructor->name->name, constructor->arguments.size())};
    if (const Signature * sig{symbol ? SignatureOf(*symbol) : nullptr}) {
      for (const Type *argument : sig->arguments) {
        MarkInvariant(argument, nullptr, base, invariant);
      }
    }
  }
  return invariants_[&base] = std::move(invariant);
}

void TypeChecker::MarkInvariant(const Type *type, const Type *context,
    const Symbol &base, std::vector<const Type *> &invariant) {
  switch (type->kind) {
  case TypeKind::Variable:
    if (context && type->var < invariant.size() && !invariant[type->var]) {
      invariant[type->var] = context;
    }
    return;
  case TypeKind::Tuple:
    // A tuple is not among the manual's places, nor mmc's.
    for (const Type *element : type->arguments) {
      MarkInvariant(element, context, base, invariant);
    }
    return;
  case TypeKind::Named:
    if (type->constructor == &base &&
        llvm::all_of(llvm::enumerate(type->arguments), [&](auto argument) {
          return argument.value() == types_.Variable(argument.index());
        })) {
      return; // base(B1, ..., Bn), recursively
    }
    [[fallthrough]];
  case TypeKind::HigherOrder:
    // In another discriminated union type, a higher-order type, an
    // abstract or foreign type: the same in both.
    for (const Type *argument : type->arguments) {
      MarkInvariant(argument, context ? context : type, base, invariant);
    }
    if (type->result) {
      MarkInvariant(type->result, context ? context : type, base, invariant);
    }
    return;
  case TypeKind::Existential:
    return;
  }
}

std::optional<std::string> TypeChecker::CoerceError(
    const Assignment &a, const Unify &unify) {
  const Type *from{Resolve(a, VarType(unify.arguments[0]))};
  const Type *to{Resolve(a, VarType(unify.left))};
  if (HasError(from) || HasError(to)) {
    return std::nullopt;
  }
  for (const Type *type : {from, to}) {
    bool free{false};
    ForEachVariable(*type, [&](unsigned var) { free |= var >= rigid_; });
    if (free) {
      return "nothing determines the type of its " +
          std::string{type == from ? "argument" : "result"} +
          ", and coerce does not";
    }
  }
  const Type *fromBase{BaseOf(from)};
  const Type *toBase{BaseOf(to)};
  for (auto [type, base] : {std::pair{from, fromBase}, {to, toBase}}) {
    if (!base) {
      return TypeText(a, type) +
          " is not a discriminated union type whose definition is visible";
    }
    if (IsError(base)) {
      return std::nullopt;
    }
  }
  if (fromBase->constructor != toBase->constructor) {
    return "their base types, " + TypeText(a, fromBase) + " and " +
        TypeText(a, toBase) + ", differ";
  }
  const Symbol &base{*fromBase->constructor};
  llvm::ArrayRef<const Type *> invariant{InvariantIn(base)};
  const auto *defn{std::get<const parser::DuTypeDefn *>(base.declaration)};
  auto parameter{[&](unsigned var) {
    return var < defn->parameters.size() ? defn->parameters[var].str()
                                         : "T" + std::to_string(var);
  }};
  for (unsigned j{0}; j < fromBase->arguments.size(); ++j) {
    const Type *s{fromBase->arguments[j]};
    const Type *t{toBase->arguments[j]};
    if (s == t) {
      continue;
    }
    std::string where{"the parameter " + parameter(j) + " of " +
        base.module.str() + "." + base.name.str() + "/" +
        std::to_string(base.arity) + ", " + TypeText(a, s) + " in one and " +
        TypeText(a, t) + " in the other,"};
    if (j < invariant.size() && invariant[j]) {
      return where + " must be the same type in both, since it occurs in " +
          TypeToString(*invariant[j], parameter) + " in its definition";
    }
    if (!IsSubtype(s, t) && !IsSubtype(t, s)) {
      return where + " and neither is a subtype of the other";
    }
  }
  return std::nullopt;
}

void TypeChecker::DefineMethods(const parser::InstanceDecl &decl,
    const Symbol &typeclass, llvm::ArrayRef<const Type *> types,
    llvm::ArrayRef<ClassConstraint> constraints,
    llvm::ArrayRef<llvm::StringRef> names, const std::string &text) {
  const ModuleSymbols *symbols{typeclass.module == resolution_.module().name()
          ? &resolution_.module()
          : resolution_.Module(typeclass.module)};
  if (!symbols) {
    return;
  }
  // The instance's variables are numbered from 0, and the method's own
  // after them.
  unsigned instanceVars(names.size());
  for (const Type *type : types) {
    ForEachVariable(*type, [&](unsigned var) {
      instanceVars = std::max(instanceVars, var + 1);
    });
  }
  for (const Symbol *method : symbols->symbols()) {
    if (method->owner != &typeclass) {
      continue;
    }
    const Signature *sig{SignatureOf(*method)};
    if (!sig || sig->constraints.empty()) {
      continue;
    }
    unsigned parameters(types.size());
    auto substitute{[&](unsigned var) {
      return var < parameters ? types[var]
                              : types_.Variable(instanceVars + var - parameters);
    }};
    llvm::SmallVector<const Type *, 4> arguments;
    for (const Type *argument : sig->arguments) {
      arguments.push_back(types_.Substitute(argument, substitute));
    }
    // The method's first constraint is its type class's, which the
    // instance itself gives.
    llvm::SmallVector<ClassConstraint, 4> all{
        constraints.begin(), constraints.end()};
    for (const ClassConstraint &constraint : sig->constraints.drop_front()) {
      all.push_back(SubstituteIn(constraint, substitute));
    }
    llvm::SmallVector<llvm::StringRef, 4> varNames{names.begin(), names.end()};
    varNames.resize(instanceVars);
    for (unsigned var{parameters}; var < sig->typeVars; ++var) {
      varNames.push_back(
          var < sig->varNames.size() ? sig->varNames[var] : llvm::StringRef{});
    }
    MethodDefinition &definition{result_.methods_.emplace_back(MethodDefinition{
        &decl, method,
        Signature{sig->kind, sig->purity,
            instanceVars + sig->typeVars - parameters,
            result_.arena_.SaveArray(llvm::ArrayRef<llvm::StringRef>{varNames}),
            result_.arena_.SaveArray(llvm::ArrayRef<const Type *>{arguments}),
            result_.arena_.SaveArray(llvm::ArrayRef<ClassConstraint>{all})},
        text})};
    // The method's existential variables stay last.
    llvm::SmallVector<ClassConstraint, 2> existential;
    for (const ClassConstraint &constraint : sig->existentialConstraints) {
      existential.push_back(SubstituteIn(constraint, substitute));
    }
    definition.signature.existentials = sig->existentials;
    definition.signature.existentialConstraints = result_.arena_.SaveArray(
        llvm::ArrayRef<ClassConstraint>{existential});
    methodDefinitions_[{&decl, method}] = &definition;
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
              // A head argument !X is two, !.X and !:X.
              for (const parser::ModedTerm *argument : clause->arguments) {
                const auto *state{
                    std::get_if<const parser::StateVar *>(&argument->term)};
                arity += state &&
                        (*state)->kind == parser::StateVarKind::Both
                    ? 2
                    : 1;
              }
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

TypeChecker::Assumed TypeChecker::AssumedBy(const Assignment &a) {
  Assumed assumed{assumed_};
  for (const ClassConstraint *given : a.given) {
    // One element, not an initializer list, so the origin is *given.
    for (const auto &entry : Assume(llvm::ArrayRef<ClassConstraint>(*given))) {
      if (llvm::none_of(assumed,
              [&](const auto &x) { return Same(x.first, entry.first); })) {
        assumed.push_back(entry);
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
      auto given{givenGoals_.find(origin)};
      return ConstraintProof{wanted, origin, nullptr, {},
          given == givenGoals_.end() ? std::nullopt
                                     : std::optional<unsigned>{given->second}};
    }
  }
  if (llvm::any_of(entailing_,
          [&](const ClassConstraint &c) { return Same(c, wanted); })) {
    return wanted; // it needs itself
  }
  // Instances do not overlap, so at most one matches.
  for (const Instance &instance : result_.instances_) {
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
        result_.arena_.SaveArray(llvm::ArrayRef<ConstraintProof>{premises}),
        std::nullopt, &instance,
        result_.arena_.SaveArray(llvm::ArrayRef<const Type *>{bindings})};
  }
  return wanted;
}

bool TypeChecker::Refuted(
    const ClassConstraint &wanted, const Assumed &assumed) {
  bool ground{true};
  for (const Type *argument : wanted.arguments) {
    ForEachVariable(*argument, [&](unsigned) { ground = false; });
    if (HasError(argument)) {
      return false;
    }
  }
  if (ground) {
    return std::holds_alternative<ClassConstraint>(Entail(wanted, assumed));
  }
  if (llvm::any_of(entailing_,
          [&](const ClassConstraint &c) { return Same(c, wanted); })) {
    return false;
  }
  for (const Instance &instance : result_.instances_) {
    llvm::SmallVector<const Type *, 4> bindings;
    if (instance.typeclass != wanted.typeclass ||
        !llvm::all_of(llvm::zip(instance.types, wanted.arguments),
            [&](auto pair) {
              return Match(std::get<0>(pair), std::get<1>(pair), bindings);
            })) {
      continue;
    }
    entailing_.push_back(wanted);
    bool refuted{llvm::any_of(instance.constraints, [&](const ClassConstraint &c) {
      return Refuted(SubstituteIn(c, [&](unsigned var) {
        return var < bindings.size() && bindings[var] ? bindings[var]
                                                      : types_.Named(error_, {});
      }), assumed);
    })};
    entailing_.pop_back();
    return refuted;
  }
  return false; // perhaps assumed, or satisfied once its types are known
}

void TypeChecker::Improve(Assignment &a) {
  // The bound variables among the first n, which a fresh variable does
  // not count among.
  auto bound{[](const Assignment &x, std::size_t n) {
    return llvm::count_if(llvm::ArrayRef<const Type *>{x.bindings}.take_front(n),
        [](const Type *t) { return t; });
  }};
  auto resolve{[&](const ClassConstraint &c) {
    return SubstituteIn(c, [&](unsigned var) { return Resolve(a, types_.Variable(var)); });
  }};
  // The wanted constraints, and those that the instances that improve
  // them need, which improve others in turn.
  std::vector<ClassConstraint> wanted;
  for (const auto &[goal, constraint] : a.wanted) {
    wanted.push_back(constraint);
  }
  // Unifies the types in a copy, which replaces the assignment if it
  // binds more of the assignment's variables. A unification that fails is left for the
  // constraint's check to report, as mmc does.
  auto attempt{[&](llvm::function_ref<bool(Assignment &)> unify) {
    Assignment copy{a};
    std::size_t n{a.bindings.size()};
    if (unify(copy) && bound(copy, n) > bound(a, n)) {
      a = std::move(copy);
      return true;
    }
    return false;
  }};
  constexpr unsigned maxRounds{64}, maxWanted{256};
  for (unsigned round{0}; round < maxRounds; ++round) {
    bool changed{false};
    for (std::size_t i{0}; i < wanted.size(); ++i) {
      ClassConstraint c{resolve(wanted[i])};
      llvm::ArrayRef<Dependency> dependencies{Dependencies(*c.typeclass)};
      if (dependencies.empty() ||
          llvm::any_of(c.arguments, [&](const Type *t) { return HasError(t); })) {
        continue;
      }
      // Another constraint of the class with the same domain, assumed or
      // wanted earlier.
      auto sameDomain{[&](const ClassConstraint &other, const Dependency &d) {
        return other.typeclass == c.typeclass &&
            llvm::all_of(d.domain, [&](unsigned position) {
              return other.arguments[position] == c.arguments[position];
            });
      }};
      auto unifyRanges{[&](const ClassConstraint &other, const Dependency &d) {
        return attempt([&](Assignment &copy) {
          return llvm::all_of(d.range, [&](unsigned position) {
            return UnifyTypes(
                copy, c.arguments[position], other.arguments[position]);
          });
        });
      }};
      for (const Dependency &d : dependencies) {
        for (const auto &[assumed, origin] : AssumedBy(a)) {
          if (sameDomain(assumed, d)) {
            changed |= unifyRanges(assumed, d);
          }
        }
        for (std::size_t k{0}; k < i; ++k) {
          ClassConstraint other{resolve(wanted[k])};
          if (sameDomain(other, d)) {
            changed |= unifyRanges(other, d);
          }
        }
        // An instance whose domain matches: its range, with its other
        // variables fresh, and its constraints are wanted too.
        for (const Instance &instance : result_.instances_) {
          llvm::SmallVector<const Type *, 4> bindings;
          if (instance.typeclass != c.typeclass ||
              !llvm::all_of(d.domain, [&](unsigned position) {
                return Match(instance.types[position], c.arguments[position],
                    bindings);
              })) {
            continue;
          }
          std::vector<ClassConstraint> premises;
          bool improved{attempt([&](Assignment &copy) {
            llvm::DenseMap<unsigned, const Type *> fresh;
            auto substitute{[&](unsigned var) {
              if (var < bindings.size() && bindings[var]) {
                return bindings[var];
              }
              auto [it, inserted]{fresh.try_emplace(var, nullptr)};
              if (inserted) {
                it->second = types_.Variable(copy.bindings.size());
                copy.bindings.push_back(nullptr);
              }
              return it->second;
            }};
            for (unsigned position : d.range) {
              if (!UnifyTypes(copy, c.arguments[position],
                      types_.Substitute(instance.types[position], substitute))) {
                return false;
              }
            }
            for (const ClassConstraint &premise : instance.constraints) {
              premises.push_back(SubstituteIn(premise, substitute));
            }
            return true;
          })};
          if (improved) {
            changed = true;
            for (const ClassConstraint &premise : premises) {
              if (wanted.size() < maxWanted) {
                wanted.push_back(premise);
              }
            }
          }
          break; // instances agree on the range, by the dependency
        }
      }
    }
    if (!changed) {
      break;
    }
  }
}

bool TypeChecker::Refuted(const Assignment &a) {
  Assumed assumed{AssumedBy(a)};
  return llvm::any_of(a.wanted, [&](const auto &wanted) {
    return Refuted(SubstituteIn(wanted.second,
        [&](unsigned var) { return Resolve(a, types_.Variable(var)); }),
        assumed);
  });
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
    return var < base_ ? rigidNames_[var].str() : "T_" + std::to_string(var);
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
    coerces_.push_back(&unify);
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

const Signature &TypeChecker::NewSignature(
    const Symbol &symbol, const Signature &sig) {
  auto [it, inserted]{newSignatures_.try_emplace(&symbol, sig)};
  if (inserted) {
    it->second.existentials = 0;
    it->second.constraints = sig.existentialConstraints;
    it->second.existentialConstraints = {};
  }
  return it->second;
}

void TypeChecker::CheckFunctor(const Unify &unify, const parser::SymName &name) {
  names_[unify.number] = &name;
  goal_ = unify.number;
  // Name resolution found existentially typed constructors for 'new f'.
  bool constructing{name.name.starts_with("new ")};
  std::vector<Option> options;
  for (const Symbol *candidate : name.candidates) {
    options.push_back([&, candidate, constructing](Assignment &a) {
      a.symbols[unify.number] = candidate;
      const Signature *sig{SignatureOf(*candidate)};
      if (sig && constructing &&
          candidate->kind == SymbolKind::Constructor) {
        sig = &NewSignature(*candidate, *sig);
      }
      if (notUpdatable_.count(candidate)) {
        return false;
      }
      // An undeclared predicate or function: its types are inferred later.
      return !sig ||
          Apply(a, *candidate, *sig, unify.arguments, VarType(unify.left));
    });
  }
  if (IsCharConstant(name, unify.arguments.size())) {
    options.push_back([&](Assignment &a) {
      return UnifyTypes(a, VarType(unify.left), Primitive("character"));
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
    for (const Symbol *candidate : name.candidates) {
      if (auto it{notUpdatable_.find(candidate)}; it != notUpdatable_.end()) {
        llvm::StringRef field{candidate->name};
        field.consume_back(" :=");
        message.Attach(Source(name.tokens),
            "field %s cannot be updated, since its existentially quantified type variable %s occurs in another field's type too"_because_en_US,
            field.str(), it->second.var.str());
        message.Attach(it->second.field, "the field is defined here"_because_en_US);
      }
    }
    for (const Symbol *candidate : name.candidates) {
      if (constructing) {
        break; // 'new f' is the universal version
      }
      const Signature *sig{SignatureOf(*candidate)};
      bool function{candidate->kind != SymbolKind::Predicate};
      if (sig && sig->existentials > 0 &&
          (!function || unify.arguments.size() + 1 < sig->arguments.size())) {
        message.Attach(Source(name.tokens),
            "%s is existentially typed, and no closure of it can be made"_because_en_US,
            SignatureToString(*candidate, *sig));
      }
    }
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
    // The closure's purity is the call's, as in mmc: P(X) needs a pure
    // closure, and impure P(X) an impure one.
    return types_.HigherOrder(call.kind, call.purity, types, result);
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

ClauseTypes TypeChecker::CheckClause(const ClauseForm &clause, bool inferring,
    const MethodDefinition *method) {
  clause_ = &clause;
  tooMany_ = false;
  failed_ = false;
  // What the clause defines, and its types, if any.
  const Symbol *definition{Definition(clause)};
  const Signature *sig{method ? &method->signature
          : definition && !inferring ? SignatureOf(*definition)
                                     : nullptr};
  base_ = sig ? sig->typeVars : 0;
  rigid_ = sig ? sig->typeVars - sig->existentials : 0;
  callExistentials_.clear();
  callGiven_.clear();
  coerces_.clear();
  rigidNames_ = sig ? sig->varNames : llvm::ArrayRef<llvm::StringRef>{};
  assumed_ = sig ? Assume(sig->constraints) : Assumed{};
  // The type qualifications' variables: the declaration's names are its
  // rigid variables, and the others are the clause's own.
  TypeVarNames names;
  for (unsigned var{0}; var < base_; ++var) {
    if (var < rigidNames_.size() && !rigidNames_[var].empty()) {
      names.numbers[rigidNames_[var]] = var;
    }
  }
  names.next = base_ + clause.vars.size();
  llvm::SmallVector<const Type *, 4> qualifications;
  for (const TypeQualification &q : clause.typeQualifications) {
    qualifications.push_back(Convert(q.type, names));
  }
  Assignment initial;
  initial.bindings.resize(names.next);
  initial.symbols.resize(clause.goals);
  const void *existentialKey{method ? static_cast<const void *>(method)
                                      : static_cast<const void *>(definition)};
  if (sig && !inferring) {
    // An earlier clause's types for the existential variables.
    auto it{existentialTypes_.find(existentialKey)};
    for (unsigned var{rigid_}; it != existentialTypes_.end() && var < base_;
         ++var) {
      initial.bindings[var] = it->second[var - rigid_];
    }
    for (const ClassConstraint &constraint : sig->existentialConstraints) {
      initial.wanted.emplace_back(clause.goals, constraint);
    }
  }
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
  for (Assignment &a : assignments_) {
    Improve(a);
  }
  if (!coerces_.empty() && assignments_.size() > 1) {
    // Assignments whose coerces cannot convert are dropped, if others are
    // left, as mmc does: so in X = k, Y = coerce(X), with Y's type aa, X
    // is aa's k and not another type's.
    std::vector<Assignment> valid;
    for (const Assignment &a : assignments_) {
      if (llvm::none_of(coerces_,
              [&](const Unify *unify) { return CoerceError(a, *unify); })) {
        valid.push_back(a);
      }
    }
    if (!valid.empty()) {
      assignments_ = std::move(valid);
    }
  }
  ClauseTypes types{Finish(clause, definition, sig)};
  types.method = method;
  if (sig && !inferring && sig->existentials > 0 && !failed_ &&
      !existentialTypes_.count(existentialKey)) {
    // The types this clause gives the existential variables, for the
    // others; one that is not determined is left to each.
    std::vector<const Type *> given;
    for (unsigned var{rigid_}; var < base_; ++var) {
      const Type *type{Resolve(assignments_.front(), types_.Variable(var))};
      bool free{false};
      ForEachVariable(*type, [&](unsigned v) { free |= v >= rigid_; });
      given.push_back(free ? nullptr : type);
    }
    existentialTypes_[existentialKey] = std::move(given);
  }
  return types;
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
            // A subtype's constructors have its supertype's names.
            if (s->owner && s->kind == SymbolKind::Constructor &&
                llvm::any_of(choices, [&](const Symbol *other) {
                  return other && other != s && other->kind == s->kind &&
                      other->module == s->module && other->name == s->name &&
                      other->arity == s->arity;
                })) {
              os << " of " << *s->owner;
            }
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
    Assumed assumed{AssumedBy(chosen)};
    llvm::SmallVector<ClassConstraint, 2> reported;
    auto name{[&](unsigned var) {
      return var < base_ ? rigidNames_[var].str() : "T_" + std::to_string(var);
    }};
    for (const auto &[goal, instantiated] : chosen.wanted) {
      ClassConstraint wanted{SubstituteIn(instantiated, [&](unsigned var) {
        return Resolve(chosen, types_.Variable(var));
      })};
      if (llvm::any_of(wanted.arguments,
              [&](const Type *t) { return HasError(t); })) {
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
      parser::CharBlock where{goal < names_.size() && names_[goal]
              ? Source(names_[goal]->tokens)
              : Source(clause.tokens)};
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
  types.typeError = failed_ || ambiguity;
  bool coerceFailed{false};
  if (sig && !ambiguity && !failed_) {
    for (const Unify *unify : coerces_) {
      if (std::optional<std::string> reason{CoerceError(chosen, *unify)}) {
        // mmc: "cannot coerce from `m.fruit' to `m.veg'".
        messages_
            ->Say(Source(unify->tokens), "cannot coerce from %s to %s"_err_en_US,
                TypeText(chosen, VarType(unify->arguments[0])),
                TypeText(chosen, VarType(unify->left)))
            .Attach(Source(unify->tokens), "%s"_because_en_US, *reason);
        coerceFailed = true;
        types.typeError = true;
      }
    }
  }
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
  if (!unbound.empty() && !coerceFailed) {
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
// name them. The types that calls give their callees' existential
// variables are its existential variables, numbered after those and named
// as the callees name them: inference infers an existential type only from
// a call (reference manual, "Semantics of type quantifiers").
Signature TypeChecker::Generalize(const Symbol &symbol, const Assignment &a,
    llvm::ArrayRef<const Type *> types,
    llvm::ArrayRef<ClassConstraint> constraints,
    llvm::ArrayRef<ClassConstraint> given) {
  llvm::DenseMap<unsigned, unsigned> numbers;
  llvm::SmallVector<const Type *, 4> arguments;
  for (const Type *type : types) {
    arguments.push_back(types_.Substitute(Resolve(a, type), [&](unsigned var) {
      auto [it, inserted]{numbers.try_emplace(var, numbers.size())};
      return types_.Variable(it->second);
    }));
  }
  unsigned universal(numbers.size());
  llvm::DenseMap<const Type *, unsigned> existentials;
  std::vector<llvm::StringRef> existentialNames;
  for (const Type *argument : arguments) {
    ForEachExistential(*argument, [&](const Type &t) {
      if (existentials.try_emplace(&t, universal + existentials.size()).second) {
        existentialNames.push_back(t.name);
      }
    });
  }
  auto replace{[&](const Type *t) {
    auto it{existentials.find(t)};
    return it == existentials.end() ? t : types_.Variable(it->second);
  }};
  for (const Type *&argument : arguments) {
    argument = types_.SubstituteExistentials(argument, replace);
  }
  std::vector<ClassConstraint> existentialConstraints;
  for (const ClassConstraint &constraint : given) {
    bool head{true};
    for (const Type *argument : constraint.arguments) {
      ForEachExistential(*argument,
          [&](const Type &t) { head &= existentials.count(&t) > 0; });
    }
    if (!head) {
      continue;
    }
    llvm::SmallVector<const Type *, 4> renamed;
    for (const Type *argument : constraint.arguments) {
      renamed.push_back(types_.SubstituteExistentials(argument, replace));
    }
    ClassConstraint c{constraint.typeclass,
        result_.arena_.SaveArray(llvm::ArrayRef<const Type *>{renamed})};
    if (llvm::none_of(existentialConstraints,
            [&](const ClassConstraint &x) { return Same(x, c); })) {
      existentialConstraints.push_back(c);
    }
  }
  // The constraints, on the arguments' variables, without repeats. The
  // clauses' unified types may make them reducible, or ground.
  std::vector<ClassConstraint> reduced;
  for (const ClassConstraint &constraint : constraints) {
    Reduce(SubstituteIn(constraint,
               [&](unsigned var) { return Resolve(a, types_.Variable(var)); }),
        reduced);
  }
  std::vector<ClassConstraint> generalized;
  for (ClassConstraint renamed : reduced) {
    // A constraint on an existential type is the call's to satisfy.
    bool determined{true};
    for (const Type *argument : renamed.arguments) {
      ForEachVariable(*argument,
          [&](unsigned var) { determined &= numbers.count(var) > 0; });
      ForEachExistential(*argument, [&](const Type &) { determined = false; });
    }
    if (!determined) {
      continue;
    }
    renamed = SubstituteIn(renamed,
        [&](unsigned var) { return types_.Variable(numbers[var]); });
    if (llvm::none_of(generalized,
            [&](const ClassConstraint &c) { return Same(c, renamed); })) {
      generalized.push_back(renamed);
    }
  }
  std::vector<llvm::StringRef> names;
  for (unsigned j{0}; j < universal; ++j) {
    names.push_back(result_.arena_.Save("T" + std::to_string(j + 1)));
  }
  for (llvm::StringRef name : existentialNames) {
    std::string unique{name.str()};
    for (unsigned k{2}; llvm::is_contained(names, unique); ++k) {
      unique = name.str() + "_" + std::to_string(k);
    }
    names.push_back(result_.arena_.Save(unique));
  }
  Signature sig{symbol.kind == SymbolKind::Function
          ? parser::PredOrFunc::Function
          : parser::PredOrFunc::Predicate,
      parser::Purity::Pure, static_cast<unsigned>(names.size()),
      result_.arena_.SaveArray(llvm::ArrayRef<llvm::StringRef>{names}),
      result_.arena_.SaveArray(llvm::ArrayRef<const Type *>{arguments}),
      result_.arena_.SaveArray(llvm::ArrayRef<ClassConstraint>{generalized})};
  sig.existentials = existentials.size();
  sig.existentialConstraints = result_.arena_.SaveArray(
      llvm::ArrayRef<ClassConstraint>{existentialConstraints});
  return sig;
}

bool TypeChecker::Reduce(
    const ClassConstraint &wanted, std::vector<ClassConstraint> &residual) {
  if (llvm::any_of(entailing_,
          [&](const ClassConstraint &c) { return Same(c, wanted); })) {
    return false;
  }
  for (const Instance &instance : result_.instances_) {
    llvm::SmallVector<const Type *, 4> bindings;
    if (instance.typeclass != wanted.typeclass ||
        !llvm::all_of(llvm::zip(instance.types, wanted.arguments),
            [&](auto pair) {
              return Match(std::get<0>(pair), std::get<1>(pair), bindings);
            })) {
      continue;
    }
    entailing_.push_back(wanted);
    bool reduced{true};
    for (const ClassConstraint &constraint : instance.constraints) {
      reduced &= Reduce(SubstituteIn(constraint,
                            [&](unsigned var) {
                              return var < bindings.size() && bindings[var]
                                  ? bindings[var]
                                  : types_.Named(error_, {});
                            }),
          residual);
    }
    entailing_.pop_back();
    return reduced;
  }
  // No instance: one with a variable is assumed, and a ground one fails.
  bool variable{false};
  for (const Type *argument : wanted.arguments) {
    ForEachVariable(*argument, [&](unsigned) { variable = true; });
  }
  if (variable &&
      llvm::none_of(residual,
          [&](const ClassConstraint &c) { return Same(c, wanted); })) {
    residual.push_back(wanted);
  }
  return variable;
}

std::vector<ClassConstraint> TypeChecker::Residual(
    const Assignment &a, llvm::DenseMap<unsigned, unsigned> &numbers) {
  std::vector<ClassConstraint> residual;
  for (const auto &[goal, instantiated] : a.wanted) {
    ClassConstraint wanted{SubstituteIn(instantiated,
        [&](unsigned var) { return Resolve(a, types_.Variable(var)); })};
    if (llvm::none_of(wanted.arguments,
            [&](const Type *t) { return HasError(t); })) {
      Reduce(wanted, residual); // a ground one fails when checked
    }
  }
  // Those whose variables are all the head's, renumbered as it is; the
  // others are reported when the clause is checked by its signature.
  std::vector<ClassConstraint> result;
  for (const ClassConstraint &constraint : residual) {
    bool determined{true};
    for (const Type *argument : constraint.arguments) {
      ForEachVariable(*argument,
          [&](unsigned var) { determined &= numbers.count(var) > 0; });
    }
    if (determined) {
      result.push_back(SubstituteIn(constraint,
          [&](unsigned var) { return types_.Variable(numbers[var]); }));
    }
  }
  return result;
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
      x->var != y->var || x->constructor != y->constructor || x->predOrFunc != y->predOrFunc ||
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
    std::vector<ClassConstraint> constraints; // in combined's variables
    std::vector<ClassConstraint> given; // on existential types
  };
  // A clause's typing: its head's types and the constraints its goals
  // need on them.
  struct Typing {
    std::vector<const Type *> head;
    std::vector<ClassConstraint> constraints;
    std::vector<ClassConstraint> given;
    bool operator==(const Typing &that) const {
      auto same{[](const std::vector<ClassConstraint> &x,
                    const std::vector<ClassConstraint> &y) {
        return x.size() == y.size() &&
            llvm::all_of(llvm::zip(x, y), [](auto pair) {
              return Same(std::get<0>(pair), std::get<1>(pair));
            });
      }};
      return head == that.head && same(constraints, that.constraints) &&
          same(given, that.given);
    }
  };
  constexpr std::size_t maxCandidates{64};
  std::vector<Candidate> candidates;
  for (const ClauseForm *clause : clauses) {
    CheckClause(*clause, true);
    // The clause's head typings, each with its variables numbered from 0
    // in the order they occur, so that equal ones are the same.
    std::vector<Typing> typings;
    unsigned count{0};
    for (const Assignment &a : assignments_) {
      llvm::DenseMap<unsigned, unsigned> numbers;
      Typing typing;
      for (VarId var : clause->head) {
        typing.head.push_back(types_.Substitute(
            Resolve(a, VarType(var)), [&](unsigned v) {
              auto [it, inserted]{numbers.try_emplace(v, numbers.size())};
              return types_.Variable(it->second);
            }));
      }
      typing.constraints = Residual(a, numbers);
      for (const ClassConstraint *given : a.given) {
        typing.given.push_back(*given);
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
      for (Typing &typing : typings) {
        Assignment combined;
        combined.bindings.resize(count);
        candidates.push_back({std::move(combined), std::move(typing.head),
            std::move(typing.constraints), std::move(typing.given)});
      }
      continue;
    }
    std::vector<Candidate> next, refuted;
    for (const Candidate &candidate : candidates) {
      for (const Typing &typing : typings) {
        if (next.size() == maxCandidates) {
          break;
        }
        Candidate trial{candidate};
        unsigned offset(trial.combined.bindings.size());
        trial.combined.bindings.resize(offset + count);
        auto rename{[&](unsigned v) { return types_.Variable(offset + v); }};
        bool fits{trial.head.size() == typing.head.size()};
        for (std::size_t j{0}; fits && j < typing.head.size(); ++j) {
          const Type *renamed{types_.Substitute(typing.head[j], rename)};
          fits = UnifyTypes(trial.combined, trial.head[j], renamed);
        }
        for (const ClassConstraint &constraint : typing.constraints) {
          trial.constraints.push_back(SubstituteIn(constraint, rename));
        }
        trial.given.insert(
            trial.given.end(), typing.given.begin(), typing.given.end());
        // As in a clause, a combination whose constraints cannot be
        // satisfied is dropped if others are left: one clause's constraint
        // may decide another's overloading.
        if (fits &&
            llvm::any_of(trial.constraints, [&](const ClassConstraint &c) {
              return Refuted(SubstituteIn(c, [&](unsigned var) {
                return Resolve(trial.combined, types_.Variable(var));
              }));
            })) {
          refuted.push_back(std::move(trial));
        } else if (fits) {
          next.push_back(std::move(trial));
        }
      }
    }
    if (next.empty()) {
      next = std::move(refuted);
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
    Signature signature{Generalize(symbol, candidate.combined, candidate.head,
        candidate.constraints, candidate.given)};
    if (!llvm::any_of(most, [&](const Signature &s) {
          return s.arguments == signature.arguments &&
              AsSet(s.constraints) == AsSet(signature.constraints) &&
              s.existentials == signature.existentials &&
              AsSet(s.existentialConstraints) ==
              AsSet(signature.existentialConstraints);
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
            next.arguments != last.arguments ||
            AsSet(next.constraints) != AsSet(last.constraints) ||
            next.existentials != last.existentials ||
            AsSet(next.existentialConstraints) !=
                AsSet(last.existentialConstraints)) {
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
  CheckTypeDefinitions();
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
  // The instances' method clauses, of the methods their type classes have,
  // in instances without errors.
  for (const MethodClause &method : form.methods) {
    const Symbol *definition{Definition(*method.clause)};
    auto it{methodDefinitions_.find({method.instance, definition})};
    if (definition && it != methodDefinitions_.end()) {
      result_.clauses_.push_back(
          CheckClause(*method.clause, false, it->second));
    }
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
