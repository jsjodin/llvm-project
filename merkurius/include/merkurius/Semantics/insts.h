//===-- include/merkurius/Semantics/insts.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_INSTS_H_
#define MERKURIUS_SEMANTICS_INSTS_H_

// Insts (reference manual, "Insts, modes, and mode definitions"), interned,
// so that equal insts are the same pointer, and the operations of mode
// analysis on them (Overton, "Precise and Expressive Mode Systems for Typed
// Logic Programming Languages", 2003, sections 3.1 and 3.2): the
// instantiatedness and matches partial orders, merging and abstract
// unification. Every inst but free carries a uniqueness (section 3.2.3).
// A named inst stays named, and is expanded one level when an operation
// needs its body, so that recursive insts such as list_skel(I) are finite.
// The operations follow the type of the variable the inst describes, which
// says what `ground` and `any` stand for.

#include "merkurius/Common/arena.h"
#include "merkurius/Common/enum-class.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Semantics/symbols.h"
#include "merkurius/Semantics/types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace merkurius::semantics {

// From the most to the least unique, the total order of section 3.2.3.
ENUM_CLASS(Uniqueness, Unique, MostlyUnique, Shared, MostlyClobbered, Clobbered)

// A Bound inst with no functors is not_reached. A Defined inst is a named
// inst applied to arguments, or, without a symbol, one an operation made
// for a recursive result. A Parameter is an inst variable of a mode
// declaration, ω =< ι; a Constrained inst is a variable's inst that is
// known to be compatible with every inst that each of a set of them
// stands for (Overton, section 4.2.3, constrained_inst(Ω, ι)). A Parameter
// is the Constrained inst of itself and its constraint.
ENUM_CLASS(InstKind, Free, Any, Ground, Bound, HigherOrder, Parameter,
    Constrained, Defined)

struct Inst;

// A function symbol of a bound inst, by name: which constructor it is
// depends on the type the inst is applied to, as in mmc. A literal's name
// is its canonical text.
struct InstFunctor {
  llvm::StringRef module; // its qualifier as written, or empty
  llvm::StringRef name;
  unsigned arity;
  bool literal;
  auto Key() const { return std::make_tuple(name, arity, literal, module); }
  bool operator==(const InstFunctor &that) const { return Key() == that.Key(); }
  bool operator<(const InstFunctor &that) const { return Key() < that.Key(); }
};

struct BoundAlternative {
  InstFunctor functor;
  llvm::ArrayRef<const Inst *> arguments;
};

// A mode of one argument, Initial >> Final.
struct ArgMode {
  const Inst *initial;
  const Inst *final;
  bool operator==(const ArgMode &that) const {
    return initial == that.initial && final == that.final;
  }
};

struct Inst {
  InstKind kind;
  // For Any, Ground, Bound and HigherOrder.
  Uniqueness uniqueness{Uniqueness::Shared};
  // For Bound, sorted by functor, each functor once.
  llvm::ArrayRef<BoundAlternative> functors{};
  // For HigherOrder: pred(Modes) is Det, or func(Modes) = Mode is Det with
  // the result's mode last; `any` for any_pred and any_func.
  parser::PredOrFunc predOrFunc{parser::PredOrFunc::Predicate};
  bool any{false};
  llvm::ArrayRef<ArgMode> modes{};
  parser::Determinism determinism{parser::Determinism::Det};
  // For a Parameter, its number and name, and the inst it must be
  // compatible with, if constrained; for a Constrained inst, ι; for a
  // Defined inst without a symbol, its number.
  unsigned number{0};
  llvm::StringRef name{};
  const Inst *constraint{nullptr};
  // For a Parameter of a declaration, which one, so that those of a callee
  // are not a caller's.
  unsigned scope{0};
  // For a Defined inst: the inst definition and its arguments; for a
  // Constrained inst, Ω, the Parameters, sorted by name.
  const Symbol *symbol{nullptr};
  llvm::ArrayRef<const Inst *> arguments{};
};

class InstContext {
public:
  const Inst *Free();
  const Inst *Any(Uniqueness = Uniqueness::Shared);
  const Inst *Ground(Uniqueness = Uniqueness::Shared);
  const Inst *NotReached();
  // Sorts the alternatives; a functor given twice keeps its first.
  const Inst *Bound(Uniqueness, llvm::ArrayRef<BoundAlternative>);
  const Inst *HigherOrder(parser::PredOrFunc, bool any,
      llvm::ArrayRef<ArgMode> modes, parser::Determinism,
      Uniqueness = Uniqueness::Shared);
  const Inst *Parameter(unsigned number, llvm::StringRef name,
      const Inst *constraint, unsigned scope = 0);
  // constrained_inst(Ω, ι): ι itself if Ω is empty, and the Parameter if
  // it is one with ι for its constraint.
  const Inst *Constrained(
      llvm::ArrayRef<const Inst *> parameters, const Inst *inner);
  const Inst *Defined(const Symbol &, llvm::ArrayRef<const Inst *> arguments);
  // An inst whose body is given later, by Define.
  const Inst *Generated();
  void Define(const Inst *generated, const Inst *body);

  // The body of the inst definition `symbol`, whose parameters are the
  // Parameters numbered from 0.
  void SetBody(const Symbol &symbol, const Inst *body) { bodies_[&symbol] = body; }
  const Inst *Body(const Symbol &symbol) const { return bodies_.lookup(&symbol); }
  // A Defined inst's body, with its arguments for its parameters; any
  // other inst itself. An inst definition without a body expands to
  // ground, so that it draws no more errors.
  const Inst *Expand(const Inst *);
  // The inst with each Parameter replaced by `replace`'s inst for it.
  const Inst *Substitute(
      const Inst *, llvm::function_ref<const Inst *(const Inst &)> replace);
  // The inst with the uniqueness of each of its parts at most `u`.
  const Inst *Share(const Inst *, Uniqueness u = Uniqueness::Shared);

private:
  const Inst *Intern(const Inst &);

  common::Arena arena_;
  // Names, once each, so that insts with equal names are equal.
  llvm::StringSet<> strings_;
  std::map<std::vector<std::uintptr_t>, const Inst *> interned_;
  llvm::DenseMap<const Symbol *, const Inst *> bodies_;
  std::vector<const Inst *> generated_;
};

// Prints an inst as mmc's messages do: free, ground, unique,
// bound(a ; f(ground)), unique(a), (pred(in, out) is det), or a named inst
// module-qualified, list.list_skel(free). Argument modes print by their
// standard names where they have one.
void PrintInst(llvm::raw_ostream &, const Inst &);
// det, semidet, cc_multi...
const char *DeterminismName(parser::Determinism);
std::string InstToString(const Inst &);
void PrintArgMode(llvm::raw_ostream &, const ArgMode &);
std::string ArgModeToString(const ArgMode &);

// The constructors of a type, which the inst operations need to expand
// ground and any into the functors they stand for.
struct ConstructorTypes {
  InstFunctor functor;
  llvm::ArrayRef<const Type *> arguments;
};
class InstTypes {
public:
  virtual ~InstTypes();
  // False if the values of `type` are not a known set of constructors: a
  // primitive, abstract, higher-order or variable type, or none (null).
  virtual bool Constructors(
      const Type *type, std::vector<ConstructorTypes> &) = 0;
};

// Where coerce cannot convert a term: the functors and argument numbers
// from the term down to the subterm, the subterm's types and inst, and
// the functor of it that its new type lacks, if that is why.
struct CoerceFailure {
  std::vector<std::pair<InstFunctor, unsigned>> path;
  const Type *from{nullptr};
  const Type *to{nullptr};
  const Inst *inst{nullptr};
  std::optional<InstFunctor> functor;
};

// The operations of mode analysis on the insts of variables of one type
// each. Recursive insts are compared coinductively, and a recursive result
// is a Defined inst without a symbol.
class InstOps {
public:
  InstOps(InstContext &insts, InstTypes &types) : insts_{insts}, types_{types} {}

  // a ⊑ b: a matches b, so a value with inst a may be passed where b is
  // expected (definition 3.2.15). A function without a higher-order inst
  // is taken to have the default one, so a higher-order function inst
  // matches ground only if it is the default (reference manual, "Default
  // insts for functions").
  bool Matches(const Inst *a, const Inst *b, const Type *type);
  // Whether a value of a type whose values are copied, not referred to,
  // matches a unique inst whatever its own uniqueness, unless clobbered;
  // mmc's matching (-fmmc-uniqueness) compares them as any other.
  void SetCopiesUnique(bool copies) { copiesUnique_ = copies; }
  // Whether the values of `type` are copied, not referred to: a number or
  // a character, or a type whose constructors are all constants.
  bool Copied(const Type *type);
  // a ≼ b: a is at least as instantiated as b, so a variable may go from
  // b to a on forward execution (definition 3.2.14).
  bool Instantiated(const Inst *a, const Inst *b, const Type *type);
  // The least upper bound under ⊑, the inst after a branched goal whose
  // branches give a and b (definition 3.1.16); null if there is none.
  const Inst *Merge(const Inst *a, const Inst *b, const Type *type);
  // The greatest lower bound under ≼, the inst of both sides after
  // unifying values with insts a and b (definition 3.2.19, without the
  // liveness and clobbering conditions, which are the caller's); null if
  // there is none, as for two higher-order values.
  const Inst *Unify(const Inst *a, const Inst *b, const Type *type);

  // The default inst of a function of `arity` arguments, func(in, ...,
  // in) = out is det.
  const Inst *DefaultFunction(std::size_t arity);
  bool IsDefaultFunction(const Inst *);

  // The insts each Parameter of `modes` stands for, from the insts of the
  // arguments passed to them (Overton, section 4.2.4), and `modes` with
  // them in place: each is bound where it first occurs in an argument's
  // initial inst, to the greatest lower bound of the argument's inst there
  // and its constraint, if the former matches the latter, merged with
  // its other occurrences in that argument, as in mmc, unless the
  // argument's inst there is an inst variable narrowed by a unification. A
  // Parameter not bound stays, as do the caller's own, `fixed`.
  std::vector<ArgMode> Instantiate(llvm::ArrayRef<ArgMode> modes,
      llvm::ArrayRef<const Inst *> actual, llvm::ArrayRef<const Type *> types,
      llvm::ArrayRef<const Inst *> fixed = {});
  // An inst without its inst variables, and expanded: a Parameter's
  // constraint, a Constrained inst's ι.
  const Inst *Plain(const Inst *);

  // The inst of coerce(X), of type `to`, for X of type `from` with inst
  // `inst` (reference manual, "Type conversions"), or null, with where
  // it fails in `failure`. `subtype` is the type checker's relation.
  const Inst *Coerce(const Inst *inst, const Type *from, const Type *to,
      llvm::function_ref<bool(const Type *, const Type *)> subtype,
      CoerceFailure &failure);

  // normalise(ι) (Overton, definition 3.4.1): a ground inst is ground of
  // the least unique uniqueness of its parts; another is kept, since mmc
  // does not normalise the arguments of a bound inst that is not ground,
  // nor an inst variable's.
  const Inst *Normalise(const Inst *, const Type *type);

  // Whether an inst is ground, or contains free, or any.
  bool IsGround(const Inst *, const Type *type);
  bool HasFree(const Inst *, const Type *type);
  bool HasAny(const Inst *, const Type *type);
  bool IsFree(const Inst *);
  bool IsNotReached(const Inst *);
  // The insts of a functor's arguments in a value with inst `inst`; none
  // if the value cannot have that functor.
  std::optional<std::vector<const Inst *>> Arguments(
      const Inst *, const InstFunctor &, const Type *type);
  // A type's constructors, cached; null if it has no known set.
  const std::vector<ConstructorTypes> *ConstructorsOf(const Type *);

private:
  // Whether an inst has a part of the given kind.
  bool Contains(const Inst *, const Type *, InstKind);
  // The types of a functor's arguments for `type`, or null types if it is
  // not one of the type's.
  std::vector<const Type *> ArgumentTypes(const InstFunctor &, const Type *);
  // ground(u) or any(u) as the Bound inst over every constructor of
  // `type`; null if it has no known set.
  const Inst *AsBound(const Inst *, const Type *);
  // Defined insts expanded.
  const Inst *Resolve(const Inst *);
  // An inst as constrained_inst(Ω, ι), with Ω empty if it has no inst
  // variables.
  std::pair<std::vector<const Inst *>, const Inst *> Split(const Inst *);
  // ι ⊑̃ ι′: ι is a sub-inst of ι′, which it matches with the same
  // uniqueness (definition 4.2.1).
  bool SubInst(const Inst *a, const Inst *b, const Type *type);
  void Bind(const Inst *actual, const Inst *initial, const Type *type,
      llvm::ArrayRef<const Inst *> fixed,
      std::map<const Inst *, const Inst *> &bound,
      std::set<std::pair<const Inst *, const Inst *>> &seen);
  bool MatchesNow(const Inst *a, const Inst *b, const Type *type);
  bool InstantiatedNow(const Inst *a, const Inst *b, const Type *type);
  const Inst *MergeNow(const Inst *a, const Inst *b, const Type *type);
  const Inst *UnifyNow(const Inst *a, const Inst *b, const Type *type);
  const Inst *Bound(Uniqueness, llvm::ArrayRef<BoundAlternative>);

  using Key = std::tuple<const Inst *, const Inst *, const Type *>;
  // Pairs assumed to be related while they are being compared.
  std::map<Key, bool> matching_, instantiating_;
  // A merge or unification in progress, and the Generated inst its
  // recursive uses refer to, if any.
  std::map<Key, const Inst *> merging_, unifying_;
  std::map<Key, const Inst *> merged_, unified_;
  InstContext &insts_;
  InstTypes &types_;
  bool copiesUnique_{false};
  llvm::DenseMap<const Type *, std::optional<std::vector<ConstructorTypes>>>
      constructors_;
};

// If `name` is a builtin inst used with a number of arguments it does
// not take, as `ground(a)` or `unique(a, b)`, the numbers it takes, as
// mmc words them ("zero", "zero or one", "one").
std::optional<llvm::StringRef> WrongBuiltinInstArity(
    const parser::SymName &name, unsigned arity);

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_INSTS_H_
