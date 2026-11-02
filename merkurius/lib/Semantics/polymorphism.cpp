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

bool IsMethod(const Symbol &symbol) {
  return symbol.owner && symbol.owner->kind == SymbolKind::Typeclass;
}

} // namespace

class PolymorphismAnalyser {
public:
  PolymorphismAnalyser(TypeChecking &types, PolymorphismAnalysis &result,
      ProcedurePolymorphism &out)
      : types_{types}, result_{result}, out_{out} {}

  void Analyse() {
    const ProcedureBody &body{*out_.body};
    if (body.method) {
      out_.unchecked = PolymorphismUnchecked::Method;
      return;
    }
    const Signature *signature{types_.SignatureOf(*body.procedure->symbol)};
    if (!signature) {
      return;
    }
    if (!signature->constraints.empty()) {
      out_.unchecked = PolymorphismUnchecked::Constraints;
      return;
    }
    if (signature->existentials != 0) {
      out_.unchecked = PolymorphismUnchecked::Existential;
      return;
    }
    out_.parameters = signature->typeVars;
    out_.names = signature->varNames;
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

  // A type's type_info: the hidden argument for one of the procedure's
  // type variables, void for another, a constant for a type without the
  // procedure's type variables, as void is one, and otherwise a cell of
  // its arguments' type_infos.
  const TypeInfo *Of(const Type *type) {
    auto it{made_.find(type)};
    if (it != made_.end()) {
      return it->second;
    }
    TypeInfo info{TypeInfoKind::Constant, type};
    if (type->kind == TypeKind::Variable) {
      if (type->var < out_.parameters) {
        info.kind = TypeInfoKind::Parameter;
        info.parameter = type->var;
      } else {
        info = {TypeInfoKind::Void};
      }
    } else if (HasParameters(*type)) {
      info.kind = TypeInfoKind::Cell;
      for (const Type *argument : type->arguments) {
        info.arguments.push_back(Of(argument));
      }
      if (type->result) {
        info.arguments.push_back(Of(type->result));
      }
    }
    const TypeInfo *made{&result_.nodes_.emplace_back(std::move(info))};
    made_.emplace(type, made);
    return made;
  }

  // The type_infos a goal passes to its callee, whose arguments are
  // `arguments`, for the callee's type variables in order.
  void Pass(const Body *goal, const Procedure &callee,
      llvm::ArrayRef<BodyVar> arguments) {
    const Signature *signature{types_.SignatureOf(*callee.symbol)};
    if (!signature) {
      return;
    }
    if (IsMethod(*callee.symbol) || !signature->constraints.empty()) {
      Unchecked(PolymorphismUnchecked::Constraints);
      return;
    }
    if (signature->existentials != 0) {
      Unchecked(PolymorphismUnchecked::Existential);
      return;
    }
    if (signature->typeVars == 0) {
      return;
    }
    std::vector<const Type *> binding(signature->typeVars, nullptr);
    for (std::size_t i{0};
         i < arguments.size() && i < signature->arguments.size(); ++i) {
      Match(signature->arguments[i], TypeOf(arguments[i]), binding);
    }
    std::vector<const TypeInfo *> passed;
    for (const Type *type : binding) {
      if (!type) {
        passed.push_back(&result_.nodes_.emplace_back(
            TypeInfo{TypeInfoKind::Void}));
        continue;
      }
      if (HasExistentials(*type)) {
        Unchecked(PolymorphismUnchecked::Existential);
        return;
      }
      passed.push_back(Of(type));
    }
    out_.goals[goal] = std::move(passed);
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
            signature && signature->typeVars != 0) {
          Unchecked(PolymorphismUnchecked::Closure);
        }
        break;
      }
      if (goal->isVar && goal->unify == UnifyKind::Test) {
        const Type *type{TypeOf(goal->left)};
        if (HasExistentials(*type)) {
          Unchecked(PolymorphismUnchecked::Existential);
        } else if (HasParameters(*type)) {
          out_.goals[goal] = {Of(type)};
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
  PolymorphismAnalysis &result_;
  ProcedurePolymorphism &out_;
  std::map<const Type *, const TypeInfo *> made_;
};

std::unique_ptr<PolymorphismAnalysis> AnalysePolymorphism(
    TypeChecking &types, const DeterminismChecking &determinism) {
  auto result{std::make_unique<PolymorphismAnalysis>()};
  for (const ProcedureBody &body : determinism.bodies()) {
    ProcedurePolymorphism &out{result->procedures_.emplace_back()};
    out.body = &body;
    PolymorphismAnalyser{types, *result, out}.Analyse();
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
  auto print{[&](const TypeInfo *info) {
    switch (info->kind) {
    case TypeInfoKind::Parameter:
      os << name(info->parameter) << " (argument)";
      break;
    case TypeInfoKind::Void:
      os << "void (constant)";
      break;
    case TypeInfoKind::Constant:
      PrintType(os, *info->type, name);
      os << " (constant)";
      break;
    case TypeInfoKind::Cell:
      PrintType(os, *info->type, name);
      os << " (cell)";
      break;
    }
  }};
  if (procedure.parameters != 0) {
    os << "  type_info arguments: ";
    for (unsigned var{0}; var < procedure.parameters; ++var) {
      os << (var ? ", " : "") << name(var);
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
      for (std::size_t i{0}; i < it->second.size(); ++i) {
        os << (i ? ", " : "");
        print(it->second[i]);
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
