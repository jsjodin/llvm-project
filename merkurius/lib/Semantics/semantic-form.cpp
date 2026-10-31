//===-- lib/Semantics/semantic-form.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/semantic-form.h"
#include "merkurius/Common/idioms.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

namespace merkurius::semantics {
namespace {

// A name as Mercury writes it: quoted unless it is a plain identifier, so
// the character '0' and the integer 0 differ.
void PrintName(llvm::raw_ostream &os, const parser::SymName &name) {
  for (llvm::StringRef qualifier : name.qualifiers) {
    os << qualifier << '.';
  }
  llvm::StringRef text{name.name};
  bool plain{!text.empty() && llvm::isLower(text.front()) &&
      llvm::all_of(text, [](char c) { return llvm::isAlnum(c) || c == '_'; })};
  if (plain) {
    os << text;
    return;
  }
  os << '\'';
  for (char c : text) {
    if (c == '\'' || c == '\\') {
      os << '\\';
    }
    os << c;
  }
  os << '\'';
}

class Dumper {
public:
  Dumper(llvm::raw_ostream &os, const ClauseForm &clause, bool nonlocals)
      : os_{os}, clause_{clause}, nonlocals_{nonlocals} {}

  void Clause() {
    os_ << EnumToString(clause_.kind) << ' ';
    PrintName(os_, *clause_.name);
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
      // A lambda's argument may have the name of a variable outside it;
      // the later of the two prints with its number.
      os_ << info.name;
      if (llvm::any_of(clause_.vars.take_front(id), [&](const VarInfo &v) {
            return v.kind == VarKind::Program && v.name == info.name;
          })) {
        os_ << '$' << id;
      }
      break;
    case VarKind::Anonymous:
      os_ << "_$" << id;
      break;
    case VarKind::Fresh:
      os_ << '$' << id;
      break;
    case VarKind::State:
      os_ << info.name << '!' << info.version;
      if (llvm::any_of(clause_.vars.take_front(id), [&](const VarInfo &v) {
            return v.kind == VarKind::State && v.name == info.name &&
                v.version == info.version;
          })) {
        os_ << '$' << id;
      }
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
    number_ = number;
  }
  // Ends the line of the goal that Start began.
  void EndLine() {
    if (nonlocals_ && number_ < clause_.nonlocals.size()) {
      os_ << " ~ [";
      Vars(clause_.nonlocals[number_]);
      os_ << ']';
    }
    os_ << '\n';
  }
  // A line that names a part of the goal above it.
  void Label(llvm::StringRef label, int depth, bool newline = true) {
    for (int j{0}; j < depth; ++j) {
      os_ << "| ";
    }
    os_ << label;
    if (newline) {
      os_ << '\n';
    }
  }
  void Goal(const semantics::Goal &goal, int depth) {
    std::visit(
        common::visitors{
            [&](const Conj *conj) {
              Start(conj->parallel ? "ParConj" : "Conj", conj->number, depth);
              EndLine();
              for (const semantics::Goal &g : conj->goals) {
                Goal(g, depth + 1);
              }
            },
            [&](const Disj *disj) {
              Start("Disj", disj->number, depth);
              EndLine();
              for (const semantics::Goal &g : disj->goals) {
                Goal(g, depth + 1);
              }
            },
            [&](const IfThenElse *ite) {
              Start("IfThenElse", ite->number, depth);
              EndLine();
              Goal(ite->condition, depth + 1);
              Goal(ite->then, depth + 1);
              Goal(ite->otherwise, depth + 1);
            },
            [&](const Not *negation) {
              Start("Not", negation->number, depth);
              EndLine();
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
              EndLine();
              if (unify->functor.kind == FunctorKind::Lambda && !unify->isVar) {
                Goal(unify->functor.lambda->body, depth + 1);
              }
            },
            [&](const Call *call) {
              Start(call->event ? "Event" : "Call", call->number, depth);
              os_ << ' ';
              PrintName(os_, *call->name);
              Arguments(call->arguments);
              EndLine();
            },
            [&](const HOCall *call) {
              Start("HOCall", call->number, depth);
              if (call->purity != parser::Purity::Pure) {
                os_ << ' '
                    << (call->purity == parser::Purity::Impure ? "impure"
                                                               : "semipure");
              }
              os_ << ' ';
              llvm::ArrayRef<VarId> arguments{call->arguments};
              if (call->kind == parser::PredOrFunc::Function) {
                Var(arguments.back());
                os_ << " = ";
                arguments = arguments.drop_back();
              }
              Var(call->closure);
              Arguments(arguments);
              EndLine();
            },
            [&](const Try *t) {
              Start("Try", t->number, depth);
              if (!t->io.empty()) {
                os_ << " io [";
                Vars(t->io);
                os_ << ']';
              }
              EndLine();
              Goal(t->goal, depth + 1);
              Label("then", depth + 1);
              Goal(t->then, depth + 2);
              if (t->otherwise) {
                Label("else", depth + 1);
                Goal(*t->otherwise, depth + 2);
              }
              auto branch{[&](const Catch &c, llvm::StringRef kind) {
                Label(kind, depth + 1, false);
                os_ << ' ';
                Var(c.exception);
                os_ << '\n';
                const auto *pattern{std::get_if<const Conj *>(&c.pattern)};
                if (!pattern || !(*pattern)->goals.empty()) {
                  Label("match", depth + 2);
                  Goal(c.pattern, depth + 3);
                }
                Goal(c.goal, depth + 2);
              }};
              for (const Catch &c : t->catches) {
                branch(c, "catch");
              }
              if (t->catchAny) {
                branch(*t->catchAny, "catch_any");
              }
            },
            [&](const Atomic *atomic) {
              Start("Atomic", atomic->number, depth);
              os_ << " outer [";
              Var(atomic->outerInitial);
              os_ << ", ";
              Var(atomic->outerFinal);
              os_ << ']';
              if (!atomic->vars.empty()) {
                os_ << " vars [";
                Vars(atomic->vars);
                os_ << ']';
              }
              EndLine();
              bool first{true};
              for (const AtomicAlternative &alternative : atomic->alternatives) {
                Label(first ? "inner" : "or_else inner", depth + 1, false);
                first = false;
                os_ << " [";
                Var(alternative.innerInitial);
                os_ << ", ";
                Var(alternative.innerFinal);
                os_ << "]\n";
                Goal(alternative.goal, depth + 2);
              }
            },
            [&](const Scope *scope) {
              Start("Scope", scope->number, depth);
              os_ << ' ' << EnumToString(scope->kind);
              if (!scope->vars.empty()) {
                os_ << " [";
                Vars(scope->vars);
                os_ << ']';
              }
              EndLine();
              Goal(scope->goal, depth + 1);
            },
        },
        goal);
  }
  void Functor(const semantics::Functor &functor,
      llvm::ArrayRef<VarId> arguments) {
    switch (functor.kind) {
    case FunctorKind::Name:
      PrintName(os_, *functor.name);
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
    case FunctorKind::Lambda:
      Lambda(*functor.lambda);
      break;
    }
  }
  // pred(X::in, Y::out) is det, with the body on the lines after.
  void Lambda(const semantics::Lambda &lambda) {
    if (lambda.purity != parser::Purity::Pure) {
      os_ << llvm::StringRef{EnumToString(lambda.purity)}.lower() << ' ';
    }
    bool function{lambda.kind == parser::PredOrFunc::Function};
    os_ << (function ? "func" : "pred");
    std::size_t count{lambda.arguments.size() - (function ? 1 : 0)};
    auto argument{[&](std::size_t j) {
      Var(lambda.arguments[j]);
      if (lambda.modes[j]) {
        os_ << "::";
        Mode(*lambda.modes[j]);
      }
    }};
    if (count > 0) {
      os_ << '(';
      for (std::size_t j{0}; j < count; ++j) {
        if (j > 0) {
          os_ << ", ";
        }
        argument(j);
      }
      os_ << ')';
    }
    if (function) {
      os_ << " = (";
      argument(count);
      os_ << ')';
    }
    if (lambda.determinism) {
      os_ << " is "
          << llvm::StringRef{EnumToString(*lambda.determinism)}.lower();
    }
  }
  // A named mode by its name, and any other by its kind.
  void Mode(const parser::Mode &mode) {
    std::visit(
        common::visitors{
            [&](const parser::NamedMode *named) {
              PrintName(os_, *named->name);
              if (!named->arguments.empty()) {
                os_ << "(...)";
              }
            },
            [&](const auto *other) {
              os_ << std::remove_pointer_t<decltype(other)>::kindName;
            },
        },
        mode);
  }

  llvm::raw_ostream &os_;
  const ClauseForm &clause_;
  bool nonlocals_;
  unsigned number_{0};
};

} // namespace

void DumpSemanticForm(
    llvm::raw_ostream &os, const SemanticForm &form, bool nonlocals) {
  for (const ClauseForm *clause : form.clauses) {
    Dumper{os, *clause, nonlocals}.Clause();
  }
  for (const MethodClause &method : form.methods) {
    os << "method of instance ";
    PrintName(os, *method.instance->name);
    os << ":\n";
    Dumper{os, *method.clause, nonlocals}.Clause();
  }
}

} // namespace merkurius::semantics
