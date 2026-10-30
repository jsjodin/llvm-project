//===-- lib/Semantics/types.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/types.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <functional>

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

// Types share their parts, and can be exponentially larger as trees than
// as the graphs they are, so the walks below visit each part once.
const Type *TypeContext::Substitute(
    const Type *type, llvm::function_ref<const Type *(unsigned)> replace) {
  llvm::DenseMap<const Type *, const Type *> done;
  std::function<const Type *(const Type *)> walk{[&](const Type *t) {
    if (t->kind == TypeKind::Variable) {
      return replace(t->var);
    }
    if (auto it{done.find(t)}; it != done.end()) {
      return it->second;
    }
    llvm::SmallVector<const Type *, 4> arguments;
    bool changed{false};
    for (const Type *argument : t->arguments) {
      arguments.push_back(walk(argument));
      changed |= arguments.back() != argument;
    }
    const Type *result{t->result ? walk(t->result) : nullptr};
    changed |= result != t->result;
    const Type *substituted{t};
    if (changed) {
      Type copy{*t};
      copy.arguments = arguments;
      copy.result = result;
      substituted = Intern(copy);
    }
    done[t] = substituted;
    return substituted;
  }};
  return walk(type);
}

void ForEachVariable(const Type &type, llvm::function_ref<void(unsigned)> f) {
  llvm::DenseSet<const Type *> seen;
  std::function<void(const Type &)> walk{[&](const Type &t) {
    if (!seen.insert(&t).second) {
      return;
    }
    if (t.kind == TypeKind::Variable) {
      f(t.var);
      return;
    }
    for (const Type *argument : t.arguments) {
      walk(*argument);
    }
    if (t.result) {
      walk(*t.result);
    }
  }};
  walk(type);
}

std::size_t TreeSize(const Type &type, std::size_t limit) {
  llvm::DenseMap<const Type *, std::size_t> sizes;
  std::function<std::size_t(const Type &)> size{[&](const Type &t) {
    if (auto it{sizes.find(&t)}; it != sizes.end()) {
      return it->second;
    }
    std::size_t n{1};
    for (const Type *argument : t.arguments) {
      n = std::min(limit, n + size(*argument));
    }
    if (t.result) {
      n = std::min(limit, n + size(*t.result));
    }
    sizes[&t] = n;
    return n;
  }};
  return size(type);
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
    // The primitive character type, as mmc names it: `char` is a type
    // only where the module char is imported (see the register).
    if (type.constructor->module.empty() && type.constructor->name == "char") {
      os << "character";
    } else {
      os << type.constructor->name;
    }
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
  // A type too large to read is cut short.
  constexpr std::size_t maxSize{200};
  if (TreeSize(type, maxSize + 1) > maxSize) {
    return "<a type with more than " + std::to_string(maxSize) + " parts>";
  }
  std::string text;
  llvm::raw_string_ostream os{text};
  PrintType(os, type, name);
  return text;
}

} // namespace merkurius::semantics
