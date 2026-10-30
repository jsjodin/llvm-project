//===-- lib/Semantics/types.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/types.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace merkurius::semantics {

const Type *TypeContext::Intern(const Type &type) {
  std::vector<std::uintptr_t> key{static_cast<std::uintptr_t>(type.kind),
      type.var, reinterpret_cast<std::uintptr_t>(type.constructor),
      static_cast<std::uintptr_t>(type.predOrFunc),
      static_cast<std::uintptr_t>(type.purity),
      reinterpret_cast<std::uintptr_t>(type.result)};
  for (const Type *argument : type.arguments) {
    key.push_back(reinterpret_cast<std::uintptr_t>(argument));
  }
  const Type *&interned{interned_[key]};
  if (!interned) {
    Type copy{type};
    copy.arguments = arena_.SaveArray(type.arguments);
    interned = arena_.Make<Type>(copy);
  }
  return interned;
}

const Type *TypeContext::Variable(unsigned var) {
  return Intern(Type{TypeKind::Variable, var, nullptr, {}, {}, {}, nullptr});
}

const Type *TypeContext::Named(
    const Symbol &constructor, llvm::ArrayRef<const Type *> arguments) {
  return Intern(
      Type{TypeKind::Named, 0, &constructor, arguments, {}, {}, nullptr});
}

const Type *TypeContext::Tuple(llvm::ArrayRef<const Type *> elements) {
  return Intern(Type{TypeKind::Tuple, 0, nullptr, elements, {}, {}, nullptr});
}

const Type *TypeContext::HigherOrder(parser::PredOrFunc kind,
    parser::Purity purity, llvm::ArrayRef<const Type *> arguments,
    const Type *result) {
  return Intern(Type{
      TypeKind::HigherOrder, 0, nullptr, arguments, kind, purity, result});
}

const Type *TypeContext::Substitute(
    const Type *type, llvm::function_ref<const Type *(unsigned)> replace) {
  if (type->kind == TypeKind::Variable) {
    return replace(type->var);
  }
  llvm::SmallVector<const Type *, 4> arguments;
  bool changed{false};
  for (const Type *argument : type->arguments) {
    arguments.push_back(Substitute(argument, replace));
    changed |= arguments.back() != argument;
  }
  const Type *result{type->result ? Substitute(type->result, replace) : nullptr};
  changed |= result != type->result;
  if (!changed) {
    return type;
  }
  Type copy{*type};
  copy.arguments = arguments;
  copy.result = result;
  return Intern(copy);
}

void ForEachVariable(const Type &type, llvm::function_ref<void(unsigned)> f) {
  if (type.kind == TypeKind::Variable) {
    f(type.var);
    return;
  }
  for (const Type *argument : type.arguments) {
    ForEachVariable(*argument, f);
  }
  if (type.result) {
    ForEachVariable(*type.result, f);
  }
}

void PrintType(llvm::raw_ostream &os, const Type &type,
    llvm::function_ref<std::string(unsigned)> name) {
  auto list{[&](llvm::ArrayRef<const Type *> types) {
    llvm::interleave(
        types, os, [&](const Type *t) { PrintType(os, *t, name); }, ", ");
  }};
  switch (type.kind) {
  case TypeKind::Variable:
    if (name) {
      os << name(type.var);
    } else {
      os << 'T' << type.var;
    }
    return;
  case TypeKind::Named:
    if (!type.constructor->module.empty()) {
      os << type.constructor->module << '.';
    }
    os << type.constructor->name;
    if (!type.arguments.empty()) {
      os << '(';
      list(type.arguments);
      os << ')';
    }
    return;
  case TypeKind::Tuple:
    os << '{';
    list(type.arguments);
    os << '}';
    return;
  case TypeKind::HigherOrder: {
    bool function{type.predOrFunc == parser::PredOrFunc::Function};
    if (function) {
      os << '(';
    }
    if (type.purity == parser::Purity::Impure) {
      os << "impure ";
    } else if (type.purity == parser::Purity::Semipure) {
      os << "semipure ";
    }
    os << (function ? "func" : "pred");
    if (!type.arguments.empty()) {
      os << '(';
      list(type.arguments);
      os << ')';
    }
    if (function) {
      os << " = ";
      PrintType(os, *type.result, name);
      os << ')';
    }
    return;
  }
  }
}

std::string TypeToString(
    const Type &type, llvm::function_ref<std::string(unsigned)> name) {
  std::string text;
  llvm::raw_string_ostream os{text};
  PrintType(os, type, name);
  return text;
}

} // namespace merkurius::semantics
