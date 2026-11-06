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

namespace merkurius::semantics {

const Type *TypeContext::Intern(const Type &type) {
  std::vector<std::uintptr_t> key{static_cast<std::uintptr_t>(type.kind),
      type.var, reinterpret_cast<std::uintptr_t>(type.constructor),
      static_cast<std::uintptr_t>(type.predOrFunc),
      static_cast<std::uintptr_t>(type.purity),
      reinterpret_cast<std::uintptr_t>(type.result),
      reinterpret_cast<std::uintptr_t>(type.name.data()), type.name.size()};
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

const Type *TypeContext::Existential(unsigned number, llvm::StringRef name) {
  return Intern(Type{
      TypeKind::Existential, number, nullptr, {}, {}, {}, nullptr, name});
}

// Types share their parts, and can be exponentially larger as trees than
// as the graphs they are, so the walks below visit each part once.
const Type *TypeContext::ReplaceLeaves(const Type *t, TypeKind leaf,
    llvm::function_ref<const Type *(const Type *)> replace,
    llvm::DenseMap<const Type *, const Type *> &done) {
  if (t->kind == leaf) {
    return replace(t);
  }
  if (auto it{done.find(t)}; it != done.end()) {
    return it->second;
  }
  llvm::SmallVector<const Type *, 4> arguments;
  bool changed{false};
  for (const Type *argument : t->arguments) {
    arguments.push_back(ReplaceLeaves(argument, leaf, replace, done));
    changed |= arguments.back() != argument;
  }
  const Type *result{
      t->result ? ReplaceLeaves(t->result, leaf, replace, done) : nullptr};
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
}

const Type *TypeContext::Substitute(
    const Type *type, llvm::function_ref<const Type *(unsigned)> replace) {
  llvm::DenseMap<const Type *, const Type *> done;
  return ReplaceLeaves(
      type, TypeKind::Variable,
      [&](const Type *variable) { return replace(variable->var); }, done);
}

const Type *TypeContext::SubstituteExistentials(
    const Type *type, llvm::function_ref<const Type *(const Type *)> replace) {
  llvm::DenseMap<const Type *, const Type *> done;
  return ReplaceLeaves(type, TypeKind::Existential, replace, done);
}

namespace {
// Calls `f` once on each part of the type of kind `leaf`, in the order they
// occur, given the parts already seen, without looking inside them.
void ForEachLeaf(const Type &t, TypeKind leaf,
    llvm::function_ref<void(const Type &)> f,
    llvm::DenseSet<const Type *> &seen) {
  if (!seen.insert(&t).second) {
    return;
  }
  if (t.kind == leaf) {
    f(t);
    return;
  }
  for (const Type *argument : t.arguments) {
    ForEachLeaf(*argument, leaf, f, seen);
  }
  if (t.result) {
    ForEachLeaf(*t.result, leaf, f, seen);
  }
}
} // namespace

void ForEachExistential(
    const Type &type, llvm::function_ref<void(const Type &)> f) {
  llvm::DenseSet<const Type *> seen;
  ForEachLeaf(type, TypeKind::Existential, f, seen);
}

void ForEachVariable(const Type &type, llvm::function_ref<void(unsigned)> f) {
  llvm::DenseSet<const Type *> seen;
  ForEachLeaf(
      type, TypeKind::Variable, [&](const Type &variable) { f(variable.var); },
      seen);
}

namespace {
std::size_t TreeSize(const Type &t, std::size_t limit,
    llvm::DenseMap<const Type *, std::size_t> &sizes) {
  if (auto it{sizes.find(&t)}; it != sizes.end()) {
    return it->second;
  }
  std::size_t n{1};
  for (const Type *argument : t.arguments) {
    n = std::min(limit, n + TreeSize(*argument, limit, sizes));
  }
  if (t.result) {
    n = std::min(limit, n + TreeSize(*t.result, limit, sizes));
  }
  sizes[&t] = n;
  return n;
}
} // namespace

std::size_t TreeSize(const Type &type, std::size_t limit) {
  llvm::DenseMap<const Type *, std::size_t> sizes;
  return TreeSize(type, limit, sizes);
}

namespace {
void PrintTypes(llvm::raw_ostream &os, llvm::ArrayRef<const Type *> types,
    llvm::function_ref<std::string(unsigned)> name) {
  llvm::interleave(
      types, os, [&](const Type *t) { PrintType(os, *t, name); }, ", ");
}
} // namespace

void PrintType(llvm::raw_ostream &os, const Type &type,
    llvm::function_ref<std::string(unsigned)> name) {
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
      PrintTypes(os, type.arguments, name);
      os << ')';
    }
    return;
  case TypeKind::Tuple:
    os << '{';
    PrintTypes(os, type.arguments, name);
    os << '}';
    return;
  case TypeKind::Existential:
    // As mmc prints one.
    os << "(some [" << type.name << "] " << type.name << ')';
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
      PrintTypes(os, type.arguments, name);
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
