//===-- lib/Semantics/polymorphism.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/polymorphism.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>
#include <map>

namespace merkurius::semantics {

namespace {

bool HasExistentials(const Type &type) {
  bool found{false};
  ForEachExistential(type, [&](const Type &) { found = true; });
  return found;
}

// Binds the type variables of `pattern`, a callee's argument type, to the
// parts of `actual`, the type of the argument it is passed.
void Match(const Type *pattern, const Type *actual,
    std::vector<const Type *> &binding) {
  if (!pattern || !actual) {
    return;
  }
  if (pattern->kind == TypeKind::Variable) {
    if (pattern->var < binding.size() && !binding[pattern->var]) {
      binding[pattern->var] = actual;
    }
    return;
  }
  if (pattern->kind != actual->kind ||
      pattern->arguments.size() != actual->arguments.size()) {
    return;
  }
  for (std::size_t i{0}; i < pattern->arguments.size(); ++i) {
    Match(pattern->arguments[i], actual->arguments[i], binding);
  }
  Match(pattern->result, actual->result, binding);
}

bool Same(const ClassConstraint &a, const ClassConstraint &b) {
  return a.typeclass == b.typeclass && llvm::equal(a.arguments, b.arguments);
}

// Which of `count` type variables the constraints have.
std::vector<bool> Constrained(
    unsigned count, llvm::ArrayRef<ClassConstraint> constraints) {
  std::vector<bool> constrained(count, false);
  for (const ClassConstraint &constraint : constraints) {
    for (const Type *type : constraint.arguments) {
      ForEachVariable(*type, [&](unsigned var) {
        if (var < count) {
          constrained[var] = true;
        }
      });
    }
  }
  return constrained;
}

// The parts of a type, in order: its arguments, then a function's result.
std::vector<const Type *> Parts(const Type &type) {
  std::vector<const Type *> parts{type.arguments.begin(), type.arguments.end()};
  if (type.result) {
    parts.push_back(type.result);
  }
  return parts;
}

// Binds an instance's type variables so that its type `pattern` is
// `actual`, if it can be: an instance's types are type constructors
// applied to distinct variables.
bool MatchInstance(const Type *pattern, const Type *actual,
    std::vector<const Type *> &binding) {
  if (pattern->kind == TypeKind::Variable) {
    if (pattern->var >= binding.size()) {
      binding.resize(pattern->var + 1, nullptr);
    }
    if (binding[pattern->var] && binding[pattern->var] != actual) {
      return false;
    }
    binding[pattern->var] = actual;
    return true;
  }
  if (pattern == actual) {
    return true;
  }
  if (pattern->kind != actual->kind || pattern->constructor != actual->constructor ||
      pattern->arguments.size() != actual->arguments.size() ||
      !pattern->result != !actual->result) {
    return false;
  }
  for (std::size_t i{0}; i < pattern->arguments.size(); ++i) {
    if (!MatchInstance(pattern->arguments[i], actual->arguments[i], binding)) {
      return false;
    }
  }
  return !pattern->result || MatchInstance(pattern->result, actual->result, binding);
}

bool Contains(const Type &type, unsigned var) {
  bool found{false};
  ForEachVariable(type, [&](unsigned v) { found |= v == var; });
  return found;
}

} // namespace

class PolymorphismAnalyser {
public:
  PolymorphismAnalyser(TypeChecking &types, const ModeChecking &modes,
      PolymorphismAnalysis &result, ProcedurePolymorphism &out)
      : types_{types}, modes_{modes}, result_{result}, out_{out} {}

  void Analyse() {
    const ProcedureBody &body{*out_.body};
    const Signature *signature{nullptr};
    // An instance method's hidden arguments are the instance's, for its
    // type variables, numbered first, and its constraints, first, and
    // then the method's.
    unsigned instanceVars{0};
    std::size_t instanceConstraints{0};
    if (body.method) {
      const MethodDefinition &definition{*body.method->method};
      signature = &definition.signature;
      const InstanceHead *head{nullptr};
      for (const InstanceHead &instance : types_.instances()) {
        if (instance.item->decl == definition.instance) {
          head = &instance;
        }
      }
      if (head) {
        for (const Type *type : head->types) {
          ForEachVariable(*type, [&](unsigned var) {
            instanceVars = std::max(instanceVars, var + 1);
          });
        }
        instanceConstraints = head->constraints.size();
      }
    } else {
      signature = types_.SignatureOf(*body.procedure->symbol);
    }
    if (!signature) {
      return;
    }
    if (signature->existentials != 0 ||
        !signature->existentialConstraints.empty()) {
      out_.unchecked = PolymorphismUnchecked::Existential;
      return;
    }
    out_.parameters = signature->typeVars;
    out_.names = signature->varNames;
    constraints_ = signature->constraints;
    out_.sources.assign(out_.parameters, nullptr);
    std::vector<bool> constrained{
        Constrained(out_.parameters, signature->constraints)};
    dictionaries_.assign(constraints_.size(), nullptr);
    auto hide{[&](unsigned firstVar, unsigned endVar, std::size_t first,
                  std::size_t end) {
      for (unsigned var{firstVar}; var < endVar; ++var) {
        if (!constrained[var]) {
          const TypeInfo *info{&result_.nodes_.emplace_back(TypeInfo{
              TypeInfoKind::Parameter, types_.types().Variable(var), var})};
          out_.sources[var] = info;
          out_.hidden.push_back({info, nullptr});
        }
      }
      for (std::size_t i{first}; i < end; ++i) {
        Dictionary dictionary{DictionaryKind::Parameter, constraints_[i]};
        dictionary.parameter = static_cast<unsigned>(i);
        dictionaries_[i] =
            &result_.dictionaries_.emplace_back(std::move(dictionary));
        out_.hidden.push_back({nullptr, dictionaries_[i]});
      }
    }};
    if (body.method) {
      instanceVars = std::min(instanceVars, out_.parameters);
      instanceConstraints = std::min(instanceConstraints, constraints_.size());
      hide(0, instanceVars, 0, instanceConstraints);
      hide(instanceVars, out_.parameters, instanceConstraints,
          constraints_.size());
    } else {
      hide(0, out_.parameters, 0, constraints_.size());
    }
    // A constrained type variable's type_info is in the typeclass_info of
    // the first constraint that has it, as in mmc, at the place of the
    // type that has it; if the type is not the variable, the variable's is
    // an argument of the type's. mmc takes the type's as the variable's
    // (see the register).
    for (unsigned var{0}; var < out_.parameters; ++var) {
      if (!constrained[var]) {
        continue;
      }
      for (std::size_t i{0}; i < constraints_.size() && !out_.sources[var];
           ++i) {
        const ClassConstraint &constraint{constraints_[i]};
        unsigned supers{static_cast<unsigned>(
            types_.SuperclassesOf(*constraint.typeclass).size())};
        for (std::size_t j{0}; j < constraint.arguments.size(); ++j) {
          const Type *type{constraint.arguments[j]};
          if (!Contains(*type, var)) {
            continue;
          }
          TypeInfo in{TypeInfoKind::Dictionary, type};
          in.dictionary = dictionaries_[i];
          in.index = supers + static_cast<unsigned>(j) + 1;
          const TypeInfo *info{&result_.nodes_.emplace_back(std::move(in))};
          while (type->kind != TypeKind::Variable) {
            std::vector<const Type *> parts{Parts(*type)};
            std::size_t k{0};
            while (!Contains(*parts[k], var)) {
              ++k;
            }
            type = parts[k];
            TypeInfo argument{TypeInfoKind::Argument, type};
            argument.arguments = {info};
            argument.index = static_cast<unsigned>(k) + 1;
            info = &result_.nodes_.emplace_back(std::move(argument));
          }
          out_.sources[var] = info;
          break;
        }
      }
    }
    // The goals of the procedure's clauses, by their modes, with the
    // clause and their number, for the type checker's proofs of their
    // callees' constraints.
    llvm::ArrayRef<ClauseModes> clauses{body.method
            ? llvm::ArrayRef<ClauseModes>{body.method->clauses}
            : modes_.ClausesOf(*body.procedure)};
    for (const ClauseModes &clause : clauses) {
      for (std::size_t i{0}; i < clause.goals.size(); ++i) {
        numbers_[&clause.goals[i]] = {clause.clause, static_cast<unsigned>(i)};
      }
    }
    if (body.body) {
      Walk(body.body);
    }
    if (out_.unchecked != PolymorphismUnchecked::None) {
      out_.goals.clear();
    }
  }

private:
  void Unchecked(PolymorphismUnchecked why) {
    if (out_.unchecked == PolymorphismUnchecked::None) {
      out_.unchecked = why;
    }
  }

  const Type *TypeOf(BodyVar var) const { return out_.body->vars[var].type; }

  bool HasParameters(const Type &type) const {
    bool found{false};
    ForEachVariable(type, [&](unsigned var) { found |= var < out_.parameters; });
    return found;
  }

  // A type's type_info: the procedure's own for one of its type variables,
  // void for another, a constant for a type without the procedure's type
  // variables, as void is one, and otherwise a cell of its arguments'
  // type_infos.
  const TypeInfo *Of(const Type *type) {
    if (type->kind == TypeKind::Variable && type->var < out_.parameters &&
        out_.sources[type->var]) {
      return out_.sources[type->var];
    }
    auto it{made_.find(type)};
    if (it != made_.end()) {
      return it->second;
    }
    TypeInfo info{TypeInfoKind::Constant, type};
    if (type->kind == TypeKind::Variable) {
      info = {TypeInfoKind::Void};
    } else if (HasParameters(*type)) {
      info.kind = TypeInfoKind::Cell;
      for (const Type *argument : Parts(*type)) {
        info.arguments.push_back(Of(argument));
      }
    }
    const TypeInfo *made{&result_.nodes_.emplace_back(std::move(info))};
    made_.emplace(type, made);
    return made;
  }

  // The typeclass_info for a constraint the type checker proved, or null if
  // the proof uses what is not analysed yet.
  const Dictionary *Prove(const ConstraintProof &proof) {
    if (proof.assumed) {
      if (proof.goal ||
          proof.assumed < constraints_.data() ||
          proof.assumed >= constraints_.data() + constraints_.size()) {
        // An existential constraint, of a callee or of a constructor.
        Unchecked(PolymorphismUnchecked::Existential);
        return nullptr;
      }
      const Dictionary *found{
          Derive(dictionaries_[proof.assumed - constraints_.data()],
              proof.constraint)};
      if (!found) {
        Unchecked(PolymorphismUnchecked::Constraints);
      }
      return found;
    }
    if (!proof.head) {
      Unchecked(PolymorphismUnchecked::Constraints);
      return nullptr;
    }
    Dictionary dictionary{DictionaryKind::Constant, proof.constraint};
    if (llvm::any_of(proof.constraint.arguments,
            [&](const Type *type) { return HasParameters(*type); })) {
      dictionary.kind = DictionaryKind::Instance;
      dictionary.instance = proof.head;
      std::vector<bool> constrained{Constrained(
          static_cast<unsigned>(proof.bindings.size()),
          proof.head->constraints)};
      for (std::size_t var{0}; var < proof.bindings.size(); ++var) {
        if (!constrained[var]) {
          dictionary.typeInfos.push_back(proof.bindings[var]
                  ? Of(proof.bindings[var])
                  : &result_.nodes_.emplace_back(
                        TypeInfo{TypeInfoKind::Void}));
        }
      }
      for (const ConstraintProof &premise : proof.premises) {
        const Dictionary *made{Prove(premise)};
        if (!made) {
          return nullptr;
        }
        dictionary.premises.push_back(made);
      }
      if (!Superclasses(dictionary)) {
        return nullptr;
      }
      for (const Type *type : proof.constraint.arguments) {
        dictionary.types.push_back(Of(type));
      }
    }
    return &result_.dictionaries_.emplace_back(std::move(dictionary));
  }

  // An instance's typeclass_info has those of its type class's
  // superclasses, on its types, as mmc's has, which the type checker did
  // not need to prove: they hold where the instance does.
  bool Superclasses(Dictionary &dictionary) {
    const ClassConstraint &constraint{dictionary.constraint};
    for (const ClassConstraint &super :
        types_.SuperclassesOf(*constraint.typeclass)) {
      const Dictionary *made{Entail(Instantiate(super, constraint.arguments))};
      if (!made) {
        Unchecked(PolymorphismUnchecked::Constraints);
        return false;
      }
      dictionary.supers.push_back(made);
    }
    return true;
  }

  // A superclass constraint, or an instance's, with its type variables
  // bound to `arguments`.
  ClassConstraint Instantiate(
      const ClassConstraint &constraint, llvm::ArrayRef<const Type *> arguments) {
    std::vector<const Type *> &types{result_.typeLists_.emplace_back()};
    for (const Type *type : constraint.arguments) {
      types.push_back(types_.types().Substitute(type, [&](unsigned var) {
        return var < arguments.size() && arguments[var]
            ? arguments[var]
            : types_.types().Variable(var);
      }));
    }
    return ClassConstraint{constraint.typeclass, types};
  }

  // The typeclass_info for a constraint, as the type checker proves one:
  // by one of the procedure's constraints, or a superclass of one, breadth
  // first, or by the instance that matches it.
  const Dictionary *Entail(const ClassConstraint &wanted) {
    std::vector<const Dictionary *> queue{dictionaries_.begin(), dictionaries_.end()};
    for (std::size_t next{0}; next < queue.size() && queue.size() < 256;
         ++next) {
      const Dictionary *at{queue[next]};
      if (Same(at->constraint, wanted)) {
        return at;
      }
      llvm::ArrayRef<ClassConstraint> supers{
          types_.SuperclassesOf(*at->constraint.typeclass)};
      for (std::size_t k{0}; k < supers.size(); ++k) {
        Dictionary super{DictionaryKind::Superclass,
            Instantiate(supers[k], at->constraint.arguments)};
        super.from = at;
        super.index = static_cast<unsigned>(k) + 1;
        queue.push_back(&result_.dictionaries_.emplace_back(std::move(super)));
      }
    }
    if (++depth_ > 32) {
      --depth_;
      return nullptr;
    }
    const Dictionary *made{nullptr};
    for (const InstanceHead &instance : types_.instances()) {
      std::vector<const Type *> binding;
      if (instance.typeclass != wanted.typeclass ||
          instance.types.size() != wanted.arguments.size() ||
          !llvm::all_of(llvm::zip(instance.types, wanted.arguments), [&](auto pair) {
            return MatchInstance(std::get<0>(pair), std::get<1>(pair), binding);
          })) {
        continue;
      }
      Dictionary dictionary{DictionaryKind::Constant, wanted};
      if (llvm::any_of(wanted.arguments,
              [&](const Type *type) { return HasParameters(*type); })) {
        dictionary.kind = DictionaryKind::Instance;
        dictionary.instance = &instance;
        std::vector<bool> constrained{Constrained(
            static_cast<unsigned>(binding.size()), instance.constraints)};
        for (std::size_t var{0}; var < binding.size(); ++var) {
          if (!constrained[var]) {
            dictionary.typeInfos.push_back(binding[var]
                    ? Of(binding[var])
                    : &result_.nodes_.emplace_back(TypeInfo{TypeInfoKind::Void}));
          }
        }
        bool ok{true};
        for (const ClassConstraint &premise : instance.constraints) {
          const Dictionary *proved{Entail(Instantiate(premise, binding))};
          if (!proved) {
            ok = false;
            break;
          }
          dictionary.premises.push_back(proved);
        }
        if (!ok || !Superclasses(dictionary)) {
          break;
        }
        for (const Type *type : wanted.arguments) {
          dictionary.types.push_back(Of(type));
        }
      }
      made = &result_.dictionaries_.emplace_back(std::move(dictionary));
      break;
    }
    --depth_;
    return made;
  }

  // The typeclass_info for `wanted`, `from` itself or one of its
  // superclasses', breadth first, in the order of their declarations, as
  // the type checker finds them.
  const Dictionary *Derive(const Dictionary *from, const ClassConstraint &wanted) {
    std::vector<const Dictionary *> queue{from};
    for (std::size_t next{0}; next < queue.size() && queue.size() < 256;
         ++next) {
      const Dictionary *at{queue[next]};
      if (Same(at->constraint, wanted)) {
        return at;
      }
      llvm::ArrayRef<ClassConstraint> supers{
          types_.SuperclassesOf(*at->constraint.typeclass)};
      for (std::size_t k{0}; k < supers.size(); ++k) {
        std::vector<const Type *> &arguments{result_.typeLists_.emplace_back()};
        for (const Type *type : supers[k].arguments) {
          arguments.push_back(types_.types().Substitute(type, [&](unsigned var) {
            return var < at->constraint.arguments.size()
                ? at->constraint.arguments[var]
                : types_.types().Variable(var);
          }));
        }
        Dictionary super{DictionaryKind::Superclass,
            ClassConstraint{supers[k].typeclass, arguments}};
        super.from = at;
        super.index = static_cast<unsigned>(k) + 1;
        queue.push_back(&result_.dictionaries_.emplace_back(std::move(super)));
      }
    }
    return nullptr;
  }

  // The type_infos and typeclass_infos a goal passes to its callee, whose
  // arguments are `arguments`.
  void Pass(const Body *goal, const Procedure &callee,
      llvm::ArrayRef<BodyVar> arguments) {
    const Signature *signature{types_.SignatureOf(*callee.symbol)};
    if (!signature) {
      return;
    }
    if (signature->existentials != 0 ||
        !signature->existentialConstraints.empty()) {
      Unchecked(PolymorphismUnchecked::Existential);
      return;
    }
    if (signature->typeVars == 0 && signature->constraints.empty()) {
      return;
    }
    std::vector<const Type *> binding(signature->typeVars, nullptr);
    for (std::size_t i{0};
         i < arguments.size() && i < signature->arguments.size(); ++i) {
      Match(signature->arguments[i], TypeOf(arguments[i]), binding);
    }
    std::vector<bool> constrained{
        Constrained(signature->typeVars, signature->constraints)};
    PassedInfos passed;
    for (std::size_t var{0}; var < binding.size(); ++var) {
      if (constrained[var]) {
        continue;
      }
      const Type *type{binding[var]};
      if (!type) {
        passed.typeInfos.push_back(
            &result_.nodes_.emplace_back(TypeInfo{TypeInfoKind::Void}));
        continue;
      }
      if (semantics::HasExistentials(*type)) {
        Unchecked(PolymorphismUnchecked::Existential);
        return;
      }
      passed.typeInfos.push_back(Of(type));
    }
    if (!signature->constraints.empty()) {
      // The type checker's proofs of the callee's constraints, at the
      // goal's number in its clause.
      std::vector<const ConstraintProof *> proofs;
      auto number{numbers_.find(goal->modes)};
      if (number != numbers_.end()) {
        for (const GoalConstraint &constraint :
            number->second.first->constraints) {
          if (constraint.goal == number->second.second) {
            proofs.push_back(&constraint.proof);
          }
        }
      }
      if (proofs.size() != signature->constraints.size()) {
        Unchecked(PolymorphismUnchecked::Constraints);
        return;
      }
      for (const ConstraintProof *proof : proofs) {
        if (ConstraintHasExistentials(proof->constraint)) {
          Unchecked(PolymorphismUnchecked::Existential);
          return;
        }
        const Dictionary *dictionary{Prove(*proof)};
        if (!dictionary) {
          return;
        }
        passed.dictionaries.push_back(dictionary);
      }
    }
    out_.goals[goal] = std::move(passed);
  }

  bool ConstraintHasExistentials(const ClassConstraint &constraint) const {
    return llvm::any_of(constraint.arguments,
        [](const Type *type) { return semantics::HasExistentials(*type); });
  }

  void Walk(const Body *goal) {
    switch (goal->kind) {
    case BodyKind::Call:
      if (goal->modes && goal->modes->callee) {
        Pass(goal, *goal->modes->callee, goal->arguments);
      }
      break;
    case BodyKind::Unify:
      if (goal->lambda) {
        Walk(goal->lambda);
        break;
      }
      if (goal->unify == UnifyKind::Call && goal->modes &&
          goal->modes->callee) {
        // A function application, whose result is its last argument.
        std::vector<BodyVar> arguments{goal->arguments};
        arguments.push_back(goal->left);
        Pass(goal, *goal->modes->callee, arguments);
        break;
      }
      if (goal->modes && goal->modes->callee) {
        // A closure of a predicate's or function's name.
        if (const Signature *signature{
                types_.SignatureOf(*goal->modes->callee->symbol)};
            signature &&
            (signature->typeVars != 0 || !signature->constraints.empty())) {
          Unchecked(PolymorphismUnchecked::Closure);
        }
        break;
      }
      if (goal->isVar && goal->unify == UnifyKind::Test) {
        const Type *type{TypeOf(goal->left)};
        if (semantics::HasExistentials(*type)) {
          Unchecked(PolymorphismUnchecked::Existential);
        } else if (HasParameters(*type)) {
          out_.goals[goal] = {{Of(type)}, {}};
        }
      }
      break;
    case BodyKind::Switch:
      for (const SwitchCase &c : goal->cases) {
        Walk(c.goal);
      }
      break;
    default:
      for (const Body *g : goal->goals) {
        Walk(g);
      }
      break;
    }
  }

  TypeChecking &types_;
  const ModeChecking &modes_;
  PolymorphismAnalysis &result_;
  ProcedurePolymorphism &out_;
  llvm::ArrayRef<ClassConstraint> constraints_;
  std::vector<const Dictionary *> dictionaries_; // by constraint
  std::map<const Type *, const TypeInfo *> made_;
  unsigned depth_{0};
  llvm::DenseMap<const GoalModes *, std::pair<const ClauseTypes *, unsigned>>
      numbers_;
};

std::unique_ptr<PolymorphismAnalysis> AnalysePolymorphism(TypeChecking &types,
    const ModeChecking &modes, const DeterminismChecking &determinism) {
  auto result{std::make_unique<PolymorphismAnalysis>()};
  for (const ProcedureBody &body : determinism.bodies()) {
    ProcedurePolymorphism &out{result->procedures_.emplace_back()};
    out.body = &body;
    PolymorphismAnalyser{types, modes, *result, out}.Analyse();
  }
  return result;
}

void DumpProcedurePolymorphism(llvm::raw_ostream &os,
    const ProcedurePolymorphism &procedure,
    const parser::TokenSequence &tokens,
    llvm::function_ref<std::string(parser::CharBlock)> position) {
  if (procedure.unchecked != PolymorphismUnchecked::None) {
    os << "  not analysed: " << EnumToString(procedure.unchecked) << '\n';
    return;
  }
  auto name{[&](unsigned var) -> std::string {
    if (var < procedure.parameters && var < procedure.names.size()) {
      return procedure.names[var].str();
    }
    return "void";
  }};
  auto constraint{[&](const ClassConstraint &c) {
    PrintConstraint(os, c, name);
  }};
  // T (argument), int (constant), list.list(T) (cell), void (constant),
  // T (type_info 1 of m.c(T)), T (argument 1 of list.list(T) (...)).
  std::function<void(const TypeInfo *)> print{[&](const TypeInfo *info) {
    if (info->kind == TypeInfoKind::Void) {
      os << "void (constant)";
      return;
    }
    PrintType(os, *info->type, name);
    switch (info->kind) {
    case TypeInfoKind::Parameter:
      os << " (argument)";
      break;
    case TypeInfoKind::Constant:
      os << " (constant)";
      break;
    case TypeInfoKind::Cell:
      os << " (cell)";
      break;
    case TypeInfoKind::Dictionary:
      os << " (type_info " << info->index << " of ";
      constraint(info->dictionary->constraint);
      os << ')';
      break;
    case TypeInfoKind::Argument:
      os << " (argument " << info->index << " of ";
      print(info->arguments.front());
      os << ')';
      break;
    case TypeInfoKind::Void:
      break;
    }
  }};
  // m.c(T) (argument), m.c(int) (constant), m.c(T) (superclass 1 of ...),
  // m.c(list.list(T)) (instance with m.c(T) (argument)).
  std::function<void(const Dictionary *)> printDictionary{
      [&](const Dictionary *dictionary) {
        constraint(dictionary->constraint);
        switch (dictionary->kind) {
        case DictionaryKind::Parameter:
          os << " (argument)";
          break;
        case DictionaryKind::Constant:
          os << " (constant)";
          break;
        case DictionaryKind::Superclass:
          os << " (superclass " << dictionary->index << " of ";
          printDictionary(dictionary->from);
          os << ')';
          break;
        case DictionaryKind::Instance: {
          os << " (instance";
          bool first{true};
          for (const auto *list : {&dictionary->premises, &dictionary->supers}) {
            for (const Dictionary *d : *list) {
              os << (first ? " with " : ", ");
              first = false;
              printDictionary(d);
            }
          }
          os << ')';
          break;
        }
        }
      }};
  if (!procedure.hidden.empty()) {
    os << "  hidden arguments: ";
    for (std::size_t i{0}; i < procedure.hidden.size(); ++i) {
      os << (i ? ", " : "");
      if (const TypeInfo *info{procedure.hidden[i].typeInfo}) {
        os << name(info->parameter);
      } else {
        constraint(procedure.hidden[i].dictionary->constraint);
      }
    }
    os << '\n';
  }
  // Where the type_infos of the constrained type variables come from, by
  // the hidden typeclass_info they are in, and their place in it.
  std::vector<unsigned> constrained;
  for (unsigned var{0}; var < procedure.sources.size(); ++var) {
    if (procedure.sources[var] &&
        procedure.sources[var]->kind != TypeInfoKind::Parameter) {
      constrained.push_back(var);
    }
  }
  auto root{[&](unsigned var) {
    const TypeInfo *info{procedure.sources[var]};
    while (info->kind == TypeInfoKind::Argument) {
      info = info->arguments.front();
    }
    return std::make_pair(info->dictionary->parameter, info->index);
  }};
  llvm::stable_sort(constrained,
      [&](unsigned a, unsigned b) { return root(a) < root(b); });
  if (!constrained.empty()) {
    os << "  type_infos in typeclass_infos: ";
    for (std::size_t i{0}; i < constrained.size(); ++i) {
      os << (i ? ", " : "");
      print(procedure.sources[constrained[i]]);
    }
    os << '\n';
  }
  const ProcedureBody &body{*procedure.body};
  std::function<void(const Body *)> dump{[&](const Body *goal) {
    auto it{procedure.goals.find(goal)};
    if (it != procedure.goals.end()) {
      os << "    " << position(tokens.tokens[goal->tokens.first].spelling);
      if (goal->kind == BodyKind::Call || goal->unify == UnifyKind::Call) {
        os << ' '
           << (goal->kind == BodyKind::Call ? "call " : "application ")
           << *goal->modes->callee->symbol;
      } else {
        os << " unify " << body.vars[goal->left].name << " = "
           << body.vars[goal->var].name;
      }
      os << ": ";
      bool first{true};
      for (const TypeInfo *info : it->second.typeInfos) {
        os << (first ? "" : ", ");
        first = false;
        print(info);
      }
      for (const Dictionary *dictionary : it->second.dictionaries) {
        os << (first ? "" : ", ");
        first = false;
        printDictionary(dictionary);
      }
      os << '\n';
    }
    if (goal->kind == BodyKind::Unify && goal->lambda) {
      dump(goal->lambda);
    } else if (goal->kind == BodyKind::Switch) {
      for (const SwitchCase &c : goal->cases) {
        dump(c.goal);
      }
    } else {
      for (const Body *g : goal->goals) {
        dump(g);
      }
    }
  }};
  if (body.body) {
    dump(body.body);
  }
}

} // namespace merkurius::semantics
