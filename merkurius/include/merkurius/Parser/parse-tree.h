//===-- include/merkurius/Parser/parse-tree.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_PARSER_PARSE_TREE_H_
#define MERKURIUS_PARSER_PARSE_TREE_H_

// The parse tree: one type per construct of the reference manual, made by
// the converter from the terms of a file's items. Nodes are immutable and
// live in a ParseTreeArena, so they are trivially destructible and refer to
// each other by pointer. A category with alternatives, such as Item, is a
// std::variant of pointers to its forms.
//
// Every node has a kindName, its first and last token, and a ForEachField
// member that calls f(name, value) for each of its other fields; Walk and
// DumpParseTree use it, so a node type needs nothing else.

#include "merkurius/Common/arena.h"
#include "merkurius/Common/enum-class.h"
#include "merkurius/Parser/term.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <type_traits>
#include <variant>

namespace merkurius::parser {

using ParseTreeArena = common::Arena;

// A name, possibly module-qualified: write, io.write or io__write.
struct SymName {
  static constexpr llvm::StringLiteral kindName{"SymName"};
  TokenRange tokens;
  llvm::ArrayRef<llvm::StringRef> qualifiers; // the outermost module first
  llvm::StringRef name;
  template <typename F> void ForEachField(F &&f) const {
    f("qualifiers", qualifiers);
    f("name", name);
  }
  // Prints io.write, with the qualifiers.
  void PrintInline(llvm::raw_ostream &) const;
};

//===----------------------------------------------------------------------===//
// The module system (reference manual, "The module system")
//===----------------------------------------------------------------------===//

// :- module Name.
struct ModuleDecl {
  static constexpr llvm::StringLiteral kindName{"ModuleDecl"};
  TokenRange tokens;
  const SymName *name;
  template <typename F> void ForEachField(F &&f) const { f("name", name); }
};

// :- end_module Name.
struct EndModuleDecl {
  static constexpr llvm::StringLiteral kindName{"EndModuleDecl"};
  TokenRange tokens;
  const SymName *name;
  template <typename F> void ForEachField(F &&f) const { f("name", name); }
};

ENUM_CLASS(Section, Interface, Implementation)

// :- interface. or :- implementation.
struct SectionDecl {
  static constexpr llvm::StringLiteral kindName{"SectionDecl"};
  TokenRange tokens;
  Section section;
  template <typename F> void ForEachField(F &&f) const {
    f("section", section);
  }
};

ENUM_CLASS(ImportKind, ImportModule, UseModule, IncludeModule)

// :- import_module Modules. and its use_module and include_module forms.
struct ImportDecl {
  static constexpr llvm::StringLiteral kindName{"ImportDecl"};
  TokenRange tokens;
  ImportKind kind;
  llvm::ArrayRef<const SymName *> modules;
  template <typename F> void ForEachField(F &&f) const {
    f("kind", kind);
    f("modules", modules);
  }
};

//===----------------------------------------------------------------------===//
// Types (reference manual, "Types")
//===----------------------------------------------------------------------===//

struct TypeVar;
struct NamedType;
struct TupleType;
struct HigherOrderType;
using Type = std::variant<const TypeVar *, const NamedType *,
    const TupleType *, const HigherOrderType *>;

// T.
struct TypeVar {
  static constexpr llvm::StringLiteral kindName{"TypeVar"};
  TokenRange tokens;
  llvm::StringRef name;
  template <typename F> void ForEachField(F &&f) const { f("name", name); }
  void PrintInline(llvm::raw_ostream &o) const { o << name; }
};

// int, list(T) or io.state: a type constructor and its arguments, which
// semantic analysis resolves, builtin types included.
struct NamedType {
  static constexpr llvm::StringLiteral kindName{"NamedType"};
  TokenRange tokens;
  const SymName *name;
  llvm::ArrayRef<Type> arguments;
  template <typename F> void ForEachField(F &&f) const {
    f("name", name);
    f("arguments", arguments);
  }
};

// {T1, T2}.
struct TupleType {
  static constexpr llvm::StringLiteral kindName{"TupleType"};
  TokenRange tokens;
  llvm::ArrayRef<Type> elements;
  template <typename F> void ForEachField(F &&f) const {
    f("elements", elements);
  }
};

//===----------------------------------------------------------------------===//
// Insts and modes (reference manual, "Modes" and "Higher-order insts and
// modes")
//===----------------------------------------------------------------------===//

struct BuiltinInst;
struct InstVar;
struct BoundInst;
struct HigherOrderInst;
struct NamedInst;
using Inst = std::variant<const BuiltinInst *, const InstVar *,
    const BoundInst *, const HigherOrderInst *, const NamedInst *>;

struct MappingMode;
struct NamedMode;
// A higher-order inst used as a mode means in(Inst).
using Mode =
    std::variant<const MappingMode *, const NamedMode *, const HigherOrderInst *>;

// The reserved inst names that take no arguments.
ENUM_CLASS(BuiltinInstKind, Free, Ground, Unique, MostlyUnique, Clobbered,
    MostlyClobbered, Any, UniqueAny, MostlyUniqueAny, ClobberedAny,
    MostlyClobberedAny, NotReached)

struct BuiltinInst {
  static constexpr llvm::StringLiteral kindName{"BuiltinInst"};
  TokenRange tokens;
  BuiltinInstKind kind;
  template <typename F> void ForEachField(F &&f) const { f("kind", kind); }
};

// I, or I =< Inst, an inst parameter constrained to be compatible with Inst.
struct InstVar {
  static constexpr llvm::StringLiteral kindName{"InstVar"};
  TokenRange tokens;
  llvm::StringRef name;
  std::optional<Inst> constraint;
  template <typename F> void ForEachField(F &&f) const {
    f("name", name);
    f("constraint", constraint);
  }
};

ENUM_CLASS(LiteralKind, Integer, Float, String)

// A number or a string. Its value is kept in canonical notation
// (PrintCanonical), as in 255, -1i8 or "a\nb"; a character code such as
// 0'a is the integer it stands for.
struct Literal {
  static constexpr llvm::StringLiteral kindName{"Literal"};
  TokenRange tokens;
  LiteralKind kind;
  llvm::StringRef text;
  template <typename F> void ForEachField(F &&f) const {
    f("kind", kind);
    f("text", text);
  }
  void PrintInline(llvm::raw_ostream &o) const { o << text; }
};

// A function symbol of a bound inst, with the insts of its arguments: f(I1,
// I2), [] or [I | J]; or a literal, which has no arguments.
struct BoundFunctor {
  static constexpr llvm::StringLiteral kindName{"BoundFunctor"};
  TokenRange tokens;
  std::variant<const SymName *, const Literal *> name;
  llvm::ArrayRef<Inst> arguments;
  template <typename F> void ForEachField(F &&f) const {
    f("name", name);
    f("arguments", arguments);
  }
};

// bound(F1 ; F2 ; ...) or bound_unique(...).
struct BoundInst {
  static constexpr llvm::StringLiteral kindName{"BoundInst"};
  TokenRange tokens;
  bool unique;
  llvm::ArrayRef<const BoundFunctor *> functors;
  template <typename F> void ForEachField(F &&f) const {
    f("unique", unique);
    f("functors", functors);
  }
};

ENUM_CLASS(PredOrFunc, Predicate, Function)

ENUM_CLASS(Determinism, Det, Semidet, Multi, Nondet, Failure, Erroneous,
    CcMulti, CcNondet)

// pred(Mode, ...) is Det or func(Mode, ...) = Mode is Det, or any_pred and
// any_func for solver types.
struct HigherOrderInst {
  static constexpr llvm::StringLiteral kindName{"HigherOrderInst"};
  TokenRange tokens;
  PredOrFunc kind;
  bool any;
  llvm::ArrayRef<Mode> arguments;
  std::optional<Mode> result; // for a function
  Determinism determinism;
  template <typename F> void ForEachField(F &&f) const {
    f("kind", kind);
    f("any", any);
    f("arguments", arguments);
    f("result", result);
    f("determinism", determinism);
  }
};

// listskel(I) or a user-defined inst.
struct NamedInst {
  static constexpr llvm::StringLiteral kindName{"NamedInst"};
  TokenRange tokens;
  const SymName *name;
  llvm::ArrayRef<Inst> arguments;
  template <typename F> void ForEachField(F &&f) const {
    f("name", name);
    f("arguments", arguments);
  }
};

// Initial >> Final.
struct MappingMode {
  static constexpr llvm::StringLiteral kindName{"MappingMode"};
  TokenRange tokens;
  Inst initial;
  Inst final;
  template <typename F> void ForEachField(F &&f) const {
    f("initial", initial);
    f("final", final);
  }
};

// in, in(I), di, or a user-defined mode.
struct NamedMode {
  static constexpr llvm::StringLiteral kindName{"NamedMode"};
  TokenRange tokens;
  const SymName *name;
  llvm::ArrayRef<Inst> arguments;
  template <typename F> void ForEachField(F &&f) const {
    f("name", name);
    f("arguments", arguments);
  }
};

//===----------------------------------------------------------------------===//
// Predicate and function declarations (reference manual, "Predicate and
// function type declarations" and "Predicate and function mode
// declarations")
//===----------------------------------------------------------------------===//

ENUM_CLASS(Purity, Pure, Semipure, Impure)

// A type, and a mode if the declaration combines the two: int::in.
struct TypeAndMode {
  static constexpr llvm::StringLiteral kindName{"TypeAndMode"};
  TokenRange tokens;
  Type type;
  std::optional<Mode> mode;
  template <typename F> void ForEachField(F &&f) const {
    f("type", type);
    f("mode", mode);
  }
};

// pred(T, ...), func(T, ...) = T, or with modes and a determinism, a
// combined higher-order type and inst (reference manual, "Combined
// higher-order types and insts"); semipure or impure.
struct HigherOrderType {
  static constexpr llvm::StringLiteral kindName{"HigherOrderType"};
  TokenRange tokens;
  Purity purity;
  PredOrFunc kind;
  llvm::ArrayRef<const TypeAndMode *> arguments;
  const TypeAndMode *result; // for a function
  std::optional<Determinism> determinism;
  template <typename F> void ForEachField(F &&f) const {
    f("purity", purity);
    f("kind", kind);
    f("arguments", arguments);
    f("result", result);
    f("determinism", determinism);
  }
};

// c(T1, T2), a type class constraint.
struct ClassConstraint {
  static constexpr llvm::StringLiteral kindName{"ClassConstraint"};
  TokenRange tokens;
  const SymName *name;
  llvm::ArrayRef<Type> arguments;
  template <typename F> void ForEachField(F &&f) const {
    f("name", name);
    f("arguments", arguments);
  }
};

// I =< Inst after '<=' (reference manual, "Constrained polymorphic modes").
struct InstConstraint {
  static constexpr llvm::StringLiteral kindName{"InstConstraint"};
  TokenRange tokens;
  llvm::StringRef var;
  Inst inst;
  template <typename F> void ForEachField(F &&f) const {
    f("var", var);
    f("inst", inst);
  }
};

using Constraint = std::variant<const ClassConstraint *, const InstConstraint *>;

// :- pred p(int::in) is det. and :- func f(T) = T., with their
// quantifiers, constraints and purity.
struct PredDecl {
  static constexpr llvm::StringLiteral kindName{"PredDecl"};
  TokenRange tokens;
  PredOrFunc kind;
  llvm::ArrayRef<llvm::StringRef> universalVars; // all [T]
  llvm::ArrayRef<llvm::StringRef> existentialVars; // some [T]
  Purity purity;
  const SymName *name;
  llvm::ArrayRef<const TypeAndMode *> arguments;
  const TypeAndMode *result; // for a function
  std::optional<Type> withType; // `with_type` Type, or : Type
  std::optional<Inst> withInst; // `with_inst` Inst
  std::optional<Determinism> determinism;
  llvm::ArrayRef<const ClassConstraint *> existentialConstraints; // =>
  llvm::ArrayRef<Constraint> universalConstraints; // <=
  template <typename F> void ForEachField(F &&f) const {
    f("kind", kind);
    f("universalVars", universalVars);
    f("existentialVars", existentialVars);
    f("purity", purity);
    f("name", name);
    f("arguments", arguments);
    f("result", result);
    f("withType", withType);
    f("withInst", withInst);
    f("determinism", determinism);
    f("existentialConstraints", existentialConstraints);
    f("universalConstraints", universalConstraints);
  }
};

// :- mode p(in, out) is det. or :- mode f(in) = out is det.
struct ModeDecl {
  static constexpr llvm::StringLiteral kindName{"ModeDecl"};
  TokenRange tokens;
  PredOrFunc kind;
  const SymName *name;
  llvm::ArrayRef<Mode> arguments;
  std::optional<Mode> result; // for a function
  std::optional<Inst> withInst;
  std::optional<Determinism> determinism;
  llvm::ArrayRef<const InstConstraint *> constraints; // <=
  template <typename F> void ForEachField(F &&f) const {
    f("kind", kind);
    f("name", name);
    f("arguments", arguments);
    f("result", result);
    f("withInst", withInst);
    f("determinism", determinism);
    f("constraints", constraints);
  }
};

//===----------------------------------------------------------------------===//
// Type, inst and mode definitions (reference manual, "User-defined types"
// and "Insts, modes, and mode definitions")
//===----------------------------------------------------------------------===//

// Name/Arity, as in direct_arg is [f/1] or for list/1.
struct NameAndArity {
  static constexpr llvm::StringLiteral kindName{"NameAndArity"};
  TokenRange tokens;
  const SymName *name;
  unsigned arity;
  template <typename F> void ForEachField(F &&f) const {
    f("name", name);
    f("arity", arity);
  }
  void PrintInline(llvm::raw_ostream &o) const {
    name->PrintInline(o);
    o << '/' << arity;
  }
};

// A constructor's argument, with a field name if it is labelled:
// age :: int.
struct ConstructorArg {
  static constexpr llvm::StringLiteral kindName{"ConstructorArg"};
  TokenRange tokens;
  const SymName *field; // null if there is no label
  Type type;
  template <typename F> void ForEachField(F &&f) const {
    f("field", field);
    f("type", type);
  }
};

// f(T1, T2), possibly existentially typed: some [T] (f(T) => c(T)).
struct Constructor {
  static constexpr llvm::StringLiteral kindName{"Constructor"};
  TokenRange tokens;
  llvm::ArrayRef<llvm::StringRef> existentialVars;
  const SymName *name;
  llvm::ArrayRef<const ConstructorArg *> arguments;
  llvm::ArrayRef<const ClassConstraint *> constraints; // =>
  template <typename F> void ForEachField(F &&f) const {
    f("existentialVars", existentialVars);
    f("name", name);
    f("arguments", arguments);
    f("constraints", constraints);
  }
};

// where equality is P, comparison is Q, direct_arg is [f/1], or where
// type_is_abstract_noncanonical (reference manual, "User-defined equality
// and comparison").
struct TypeAttributes {
  static constexpr llvm::StringLiteral kindName{"TypeAttributes"};
  TokenRange tokens;
  const SymName *equality; // null if none is given
  const SymName *comparison;
  llvm::ArrayRef<const NameAndArity *> directArgs;
  bool abstractNoncanonical;
  template <typename F> void ForEachField(F &&f) const {
    f("equality", equality);
    f("comparison", comparison);
    f("directArgs", directArgs);
    f("abstractNoncanonical", abstractNoncanonical);
  }
};

// :- type t(T) ---> Constructors., or a subtype, :- type s =< t ---> ...
struct DuTypeDefn {
  static constexpr llvm::StringLiteral kindName{"DuTypeDefn"};
  TokenRange tokens;
  const SymName *name;
  llvm::ArrayRef<llvm::StringRef> parameters;
  std::optional<Type> supertype;
  llvm::ArrayRef<const Constructor *> constructors;
  const TypeAttributes *attributes; // null if there is no where
  template <typename F> void ForEachField(F &&f) const {
    f("name", name);
    f("parameters", parameters);
    f("supertype", supertype);
    f("constructors", constructors);
    f("attributes", attributes);
  }
};

// :- type t(T) == Type.
struct EquivalenceTypeDefn {
  static constexpr llvm::StringLiteral kindName{"EquivalenceTypeDefn"};
  TokenRange tokens;
  const SymName *name;
  llvm::ArrayRef<llvm::StringRef> parameters;
  Type type;
  template <typename F> void ForEachField(F &&f) const {
    f("name", name);
    f("parameters", parameters);
    f("type", type);
  }
};

// :- type t(T)., whose definition is elsewhere.
struct AbstractTypeDecl {
  static constexpr llvm::StringLiteral kindName{"AbstractTypeDecl"};
  TokenRange tokens;
  const SymName *name;
  llvm::ArrayRef<llvm::StringRef> parameters;
  const TypeAttributes *attributes; // null if there is no where
  template <typename F> void ForEachField(F &&f) const {
    f("name", name);
    f("parameters", parameters);
    f("attributes", attributes);
  }
};

// :- inst i(I) == Inst., or :- inst i ---> Functors., which is bound(...);
// either may say which type constructor it is for.
struct InstDefn {
  static constexpr llvm::StringLiteral kindName{"InstDefn"};
  TokenRange tokens;
  const SymName *name;
  llvm::ArrayRef<llvm::StringRef> parameters;
  const NameAndArity *forType; // null if there is no for
  Inst inst;
  template <typename F> void ForEachField(F &&f) const {
    f("name", name);
    f("parameters", parameters);
    f("forType", forType);
    f("inst", inst);
  }
};

// :- mode m(I) == Mode.
struct ModeDefn {
  static constexpr llvm::StringLiteral kindName{"ModeDefn"};
  TokenRange tokens;
  const SymName *name;
  llvm::ArrayRef<llvm::StringRef> parameters;
  Mode mode;
  template <typename F> void ForEachField(F &&f) const {
    f("name", name);
    f("parameters", parameters);
    f("mode", mode);
  }
};

//===----------------------------------------------------------------------===//
// Type classes (reference manual, "Type classes")
//===----------------------------------------------------------------------===//

// A functional dependency, (A, B -> C) (reference manual, "Functional
// dependencies").
struct FunctionalDependency {
  static constexpr llvm::StringLiteral kindName{"FunctionalDependency"};
  TokenRange tokens;
  llvm::ArrayRef<llvm::StringRef> domain;
  llvm::ArrayRef<llvm::StringRef> range;
  template <typename F> void ForEachField(F &&f) const {
    f("domain", domain);
    f("range", range);
  }
};

using TypeclassConstraint =
    std::variant<const ClassConstraint *, const FunctionalDependency *>;

using ClassMethod = std::variant<const PredDecl *, const ModeDecl *>;

// :- typeclass c(T) <= Superclasses where [Methods]. Without "where [...]"
// it is abstract.
struct TypeclassDecl {
  static constexpr llvm::StringLiteral kindName{"TypeclassDecl"};
  TokenRange tokens;
  const SymName *name;
  llvm::ArrayRef<llvm::StringRef> parameters;
  llvm::ArrayRef<TypeclassConstraint> constraints;
  bool abstract;
  llvm::ArrayRef<ClassMethod> methods;
  template <typename F> void ForEachField(F &&f) const {
    f("name", name);
    f("parameters", parameters);
    f("constraints", constraints);
    f("abstract", abstract);
    f("methods", methods);
  }
};

// pred(Method/Arity) is Implementation, in an instance declaration.
struct MethodByName {
  static constexpr llvm::StringLiteral kindName{"MethodByName"};
  TokenRange tokens;
  PredOrFunc kind;
  const NameAndArity *method;
  const SymName *implementation;
  template <typename F> void ForEachField(F &&f) const {
    f("kind", kind);
    f("method", method);
    f("implementation", implementation);
  }
};

// :- instance c(Types) <= Constraints where [Methods]. Without "where
// [...]" it is abstract.
struct InstanceDecl {
  static constexpr llvm::StringLiteral kindName{"InstanceDecl"};
  TokenRange tokens;
  const SymName *name;
  llvm::ArrayRef<Type> types;
  llvm::ArrayRef<const ClassConstraint *> constraints;
  bool abstract;
  llvm::ArrayRef<const MethodByName *> methods;
  template <typename F> void ForEachField(F &&f) const {
    f("name", name);
    f("types", types);
    f("constraints", constraints);
    f("abstract", abstract);
    f("methods", methods);
  }
};

//===----------------------------------------------------------------------===//
// Items (reference manual, "Items")
//===----------------------------------------------------------------------===//

using Item = std::variant<const ModuleDecl *, const EndModuleDecl *,
    const SectionDecl *, const ImportDecl *, const PredDecl *,
    const ModeDecl *, const DuTypeDefn *, const EquivalenceTypeDefn *,
    const AbstractTypeDecl *, const InstDefn *, const ModeDefn *,
    const TypeclassDecl *, const InstanceDecl *>;

// The items of a file that converted without errors, in order.
struct ParseTree {
  static constexpr llvm::StringLiteral kindName{"ParseTree"};
  TokenRange tokens;
  llvm::ArrayRef<Item> items;
  template <typename F> void ForEachField(F &&f) const { f("items", items); }
};

//===----------------------------------------------------------------------===//
// Walking and dumping
//===----------------------------------------------------------------------===//

template <typename T, typename = void> struct IsNode : std::false_type {};
template <typename T>
struct IsNode<T, std::void_t<decltype(T::kindName)>> : std::true_type {};

template <typename T> struct IsVariant : std::false_type {};
template <typename... A>
struct IsVariant<std::variant<A...>> : std::true_type {};

template <typename T> struct IsOptional : std::false_type {};
template <typename T>
struct IsOptional<std::optional<T>> : std::true_type {};

template <typename T> struct IsArrayRef : std::false_type {};
template <typename T> struct IsArrayRef<llvm::ArrayRef<T>> : std::true_type {};

namespace detail {
template <typename V, typename N>
auto CallPre(V &visitor, const N &node, int) -> decltype(visitor.Pre(node)) {
  return visitor.Pre(node);
}
template <typename V, typename N> bool CallPre(V &, const N &, long) {
  return true;
}
template <typename V, typename N>
auto CallPost(V &visitor, const N &node, int) -> decltype(visitor.Post(node)) {
  visitor.Post(node);
}
template <typename V, typename N> void CallPost(V &, const N &, long) {}
} // namespace detail

// Visits `x` and everything below it. For each node, visitor.Pre(node) is
// called first, if the visitor has a Pre for that node type, and the node's
// fields are visited only if it returns true; then visitor.Post(node) is
// called, if there is one.
template <typename V, typename T> void Walk(const T &x, V &visitor) {
  if constexpr (IsNode<T>::value) {
    if (detail::CallPre(visitor, x, 0)) {
      x.ForEachField(
          [&](llvm::StringRef, const auto &field) { Walk(field, visitor); });
      detail::CallPost(visitor, x, 0);
    }
  } else if constexpr (std::is_pointer_v<T>) {
    if (x) {
      Walk(*x, visitor);
    }
  } else if constexpr (IsOptional<T>::value) {
    if (x) {
      Walk(*x, visitor);
    }
  } else if constexpr (IsVariant<T>::value) {
    std::visit([&](const auto &alternative) { Walk(alternative, visitor); }, x);
  } else if constexpr (IsArrayRef<T>::value) {
    for (const auto &element : x) {
      Walk(element, visitor);
    }
  }
}

// Prints the tree, one node or field per line, indented by depth.
void DumpParseTree(llvm::raw_ostream &, const ParseTree &);

} // namespace merkurius::parser
#endif // MERKURIUS_PARSER_PARSE_TREE_H_
