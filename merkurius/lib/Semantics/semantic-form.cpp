//===-- lib/Semantics/semantic-form.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/semantic-form.h"
#include "merkurius/Common/idioms.h"

namespace merkurius::semantics {
namespace {

class Dumper {
public:
  Dumper(llvm::raw_ostream &os, const ClauseForm &clause)
      : os_{os}, clause_{clause} {}

  void Clause() {
    os_ << EnumToString(clause_.kind) << ' ';
    clause_.name->PrintInline(os_);
    llvm::ArrayRef<VarId> arguments{clause_.head};
    if (clause_.kind == parser::PredOrFunc::Function) {
      arguments = arguments.drop_back();
    }
    Arguments(arguments);
    if (clause_.kind == parser::PredOrFunc::Function) {
      os_ << " = ";
      Var(clause_.head.back());
    }
    os_ << '\n';
    for (const TypeQualification &qualification : clause_.typeQualifications) {
      os_ << "| type: ";
      Var(qualification.var);
      os_ << " : ";
      Type(qualification.type);
      os_ << '\n';
    }
    Goal(clause_.body, 1);
  }

private:
  void Var(VarId id) {
    const VarInfo &info{clause_.vars[id]};
    switch (info.kind) {
    case VarKind::Program:
      os_ << info.name;
      break;
    case VarKind::Anonymous:
      os_ << "_$" << id;
      break;
    case VarKind::Fresh:
      os_ << '$' << id;
      break;
    }
  }
  void Vars(llvm::ArrayRef<VarId> vars) {
    llvm::interleave(
        vars, os_, [&](VarId id) { Var(id); }, ", ");
  }
  void Arguments(llvm::ArrayRef<VarId> vars) {
    if (!vars.empty()) {
      os_ << '(';
      Vars(vars);
      os_ << ')';
    }
  }
  // A type as written, except that a higher-order type prints as its kind.
  void Type(const parser::Type &type) {
    auto types{[&](llvm::ArrayRef<parser::Type> types) {
      llvm::interleave(
          types, os_, [&](const parser::Type &t) { Type(t); }, ", ");
    }};
    std::visit(
        common::visitors{
            [&](const parser::TypeVar *var) { var->PrintInline(os_); },
            [&](const parser::NamedType *named) {
              named->name->PrintInline(os_);
              if (!named->arguments.empty()) {
                os_ << '(';
                types(named->arguments);
                os_ << ')';
              }
            },
            [&](const parser::TupleType *tuple) {
              os_ << '{';
              types(tuple->elements);
              os_ << '}';
            },
            [&](const parser::HigherOrderType *) {
              os_ << parser::HigherOrderType::kindName;
            },
        },
        type);
  }
  void Start(llvm::StringRef kind, unsigned number, int depth) {
    for (int j{0}; j < depth; ++j) {
      os_ << "| ";
    }
    os_ << kind << " #" << number;
  }
  void Goal(const semantics::Goal &goal, int depth) {
    std::visit(
        common::visitors{
            [&](const Conj *conj) {
              Start(conj->parallel ? "ParConj" : "Conj", conj->number, depth);
              os_ << '\n';
              for (const semantics::Goal &g : conj->goals) {
                Goal(g, depth + 1);
              }
            },
            [&](const Disj *disj) {
              Start("Disj", disj->number, depth);
              os_ << '\n';
              for (const semantics::Goal &g : disj->goals) {
                Goal(g, depth + 1);
              }
            },
            [&](const IfThenElse *ite) {
              Start("IfThenElse", ite->number, depth);
              os_ << '\n';
              Goal(ite->condition, depth + 1);
              Goal(ite->then, depth + 1);
              Goal(ite->otherwise, depth + 1);
            },
            [&](const Not *negation) {
              Start("Not", negation->number, depth);
              os_ << '\n';
              Goal(negation->goal, depth + 1);
            },
            [&](const Unify *unify) {
              Start("Unify", unify->number, depth);
              os_ << ' ';
              Var(unify->left);
              os_ << " = ";
              if (unify->isVar) {
                Var(unify->var);
              } else {
                Functor(unify->functor, unify->arguments);
              }
              os_ << '\n';
            },
            [&](const Call *call) {
              Start(call->event ? "Event" : "Call", call->number, depth);
              os_ << ' ';
              call->name->PrintInline(os_);
              Arguments(call->arguments);
              os_ << '\n';
            },
            [&](const HOCall *call) {
              Start("HOCall", call->number, depth);
              os_ << ' ';
              llvm::ArrayRef<VarId> arguments{call->arguments};
              if (call->kind == parser::PredOrFunc::Function) {
                Var(arguments.back());
                os_ << " = ";
                arguments = arguments.drop_back();
              }
              Var(call->closure);
              Arguments(arguments);
              os_ << '\n';
            },
            [&](const Scope *scope) {
              Start("Scope", scope->number, depth);
              os_ << ' ' << EnumToString(scope->kind);
              if (!scope->vars.empty()) {
                os_ << " [";
                Vars(scope->vars);
                os_ << ']';
              }
              os_ << '\n';
              Goal(scope->goal, depth + 1);
            },
        },
        goal);
  }
  void Functor(const semantics::Functor &functor,
      llvm::ArrayRef<VarId> arguments) {
    switch (functor.kind) {
    case FunctorKind::Name:
      functor.name->PrintInline(os_);
      Arguments(arguments);
      break;
    case FunctorKind::Cons:
      os_ << '[';
      Var(arguments[0]);
      os_ << " | ";
      Var(arguments[1]);
      os_ << ']';
      break;
    case FunctorKind::Nil:
      os_ << "[]";
      break;
    case FunctorKind::Tuple:
      os_ << '{';
      Vars(arguments);
      os_ << '}';
      break;
    case FunctorKind::Literal:
      functor.literal->PrintInline(os_);
      break;
    case FunctorKind::ImplementationLiteral:
      functor.implementation->PrintInline(os_);
      break;
    case FunctorKind::Coerce:
      os_ << "coerce";
      Arguments(arguments);
      break;
    }
  }

  llvm::raw_ostream &os_;
  const ClauseForm &clause_;
};

} // namespace

void DumpSemanticForm(llvm::raw_ostream &os, const SemanticForm &form) {
  for (const ClauseForm *clause : form.clauses) {
    Dumper{os, *clause}.Clause();
  }
}

} // namespace merkurius::semantics
