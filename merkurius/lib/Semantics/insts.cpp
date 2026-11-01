//===-- lib/Semantics/insts.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/insts.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <deque>
#include <functional>

namespace merkurius::semantics {

namespace {
// The less unique of two uniquenesses.
Uniqueness Least(Uniqueness a, Uniqueness b) { return std::max(a, b); }

// u ⊴ v: u is at least as unique as v.
bool AtLeastAsUnique(Uniqueness u, Uniqueness v) { return u <= v; }

// Whether two functors are the same one: a qualifier written on only one
// of them does not tell them apart.
bool SameFunctor(const InstFunctor &f, const InstFunctor &g) {
  return f.name == g.name && f.arity == g.arity && f.literal == g.literal &&
      (f.module.empty() || g.module.empty() || f.module == g.module);
}

const BoundAlternative *Find(
    llvm::ArrayRef<BoundAlternative> alternatives, const InstFunctor &f) {
  for (const BoundAlternative &alternative : alternatives) {
    if (SameFunctor(alternative.functor, f)) {
      return &alternative;
    }
  }
  return nullptr;
}
} // namespace

//===----------------------------------------------------------------------===//
// InstContext
//===----------------------------------------------------------------------===//

const Inst *InstContext::Intern(const Inst &inst) {
  auto text{[&](llvm::StringRef s) {
    return s.empty() ? llvm::StringRef{} : strings_.insert(s).first->getKey();
  }};
  auto word{[](auto x) { return static_cast<std::uintptr_t>(x); }};
  auto pointer{[](const void *p) { return reinterpret_cast<std::uintptr_t>(p); }};
  Inst copy{inst};
  copy.name = text(inst.name);
  std::vector<BoundAlternative> functors;
  for (BoundAlternative alternative : inst.functors) {
    alternative.functor.module = text(alternative.functor.module);
    alternative.functor.name = text(alternative.functor.name);
    functors.push_back(alternative);
  }
  std::vector<std::uintptr_t> key{word(copy.kind), word(copy.uniqueness),
      word(copy.predOrFunc), word(copy.any), word(copy.determinism),
      copy.number, copy.scope, pointer(copy.name.data()), pointer(copy.constraint),
      pointer(copy.symbol), functors.size()};
  for (const BoundAlternative &alternative : functors) {
    key.push_back(pointer(alternative.functor.module.data()));
    key.push_back(pointer(alternative.functor.name.data()));
    key.push_back(alternative.functor.arity);
    key.push_back(alternative.functor.literal);
    for (const Inst *argument : alternative.arguments) {
      key.push_back(pointer(argument));
    }
  }
  key.push_back(copy.modes.size());
  for (const ArgMode &mode : copy.modes) {
    key.push_back(pointer(mode.initial));
    key.push_back(pointer(mode.final));
  }
  key.push_back(copy.arguments.size());
  for (const Inst *argument : copy.arguments) {
    key.push_back(pointer(argument));
  }
  const Inst *&interned{interned_[key]};
  if (!interned) {
    for (BoundAlternative &alternative : functors) {
      alternative.arguments = arena_.SaveArray(alternative.arguments);
    }
    copy.functors =
        arena_.SaveArray(llvm::ArrayRef<BoundAlternative>{functors});
    copy.modes = arena_.SaveArray(copy.modes);
    copy.arguments = arena_.SaveArray(copy.arguments);
    interned = arena_.Make<Inst>(copy);
  }
  return interned;
}

const Inst *InstContext::Free() { return Intern(Inst{InstKind::Free}); }

const Inst *InstContext::Any(Uniqueness u) {
  return Intern(Inst{InstKind::Any, u});
}

const Inst *InstContext::Ground(Uniqueness u) {
  return Intern(Inst{InstKind::Ground, u});
}

const Inst *InstContext::NotReached() {
  return Intern(Inst{InstKind::Bound, Uniqueness::Shared});
}

const Inst *InstContext::Bound(
    Uniqueness u, llvm::ArrayRef<BoundAlternative> alternatives) {
  if (alternatives.empty()) {
    return NotReached();
  }
  std::vector<BoundAlternative> sorted;
  for (const BoundAlternative &alternative : alternatives) {
    if (!llvm::any_of(sorted, [&](const BoundAlternative &a) {
          return a.functor == alternative.functor;
        })) {
      sorted.push_back(alternative);
    }
  }
  std::stable_sort(sorted.begin(), sorted.end(),
      [](const BoundAlternative &a, const BoundAlternative &b) {
        return a.functor < b.functor;
      });
  Inst inst{InstKind::Bound, u};
  inst.functors = sorted;
  return Intern(inst);
}

const Inst *InstContext::HigherOrder(parser::PredOrFunc kind, bool any,
    llvm::ArrayRef<ArgMode> modes, parser::Determinism determinism,
    Uniqueness u) {
  Inst inst{InstKind::HigherOrder, u};
  inst.predOrFunc = kind;
  inst.any = any;
  inst.modes = modes;
  inst.determinism = determinism;
  return Intern(inst);
}

const Inst *InstContext::Parameter(unsigned number, llvm::StringRef name,
    const Inst *constraint, unsigned scope) {
  Inst inst{InstKind::Parameter};
  inst.number = number;
  inst.scope = scope;
  inst.name = name;
  inst.constraint = constraint;
  return Intern(inst);
}

const Inst *InstContext::Constrained(
    llvm::ArrayRef<const Inst *> parameters, const Inst *inner) {
  if (parameters.empty()) {
    return inner;
  }
  if (parameters.size() == 1 && parameters.front()->constraint == inner) {
    return parameters.front();
  }
  std::vector<const Inst *> sorted{parameters.vec()};
  llvm::sort(sorted, [](const Inst *x, const Inst *y) {
    return std::make_tuple(x->name, x->scope, x->number) <
        std::make_tuple(y->name, y->scope, y->number);
  });
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  Inst inst{InstKind::Constrained};
  inst.constraint = inner;
  inst.arguments = sorted;
  return Intern(inst);
}

const Inst *InstContext::Defined(
    const Symbol &symbol, llvm::ArrayRef<const Inst *> arguments) {
  Inst inst{InstKind::Defined};
  inst.symbol = &symbol;
  inst.arguments = arguments;
  return Intern(inst);
}

const Inst *InstContext::Generated() {
  Inst inst{InstKind::Defined};
  inst.number = generated_.size();
  generated_.push_back(nullptr);
  return arena_.Make<Inst>(inst);
}

void InstContext::Define(const Inst *generated, const Inst *body) {
  generated_[generated->number] = body;
}

const Inst *InstContext::Expand(const Inst *inst) {
  if (inst->kind != InstKind::Defined) {
    return inst;
  }
  if (!inst->symbol) {
    const Inst *body{generated_[inst->number]};
    return body ? body : Ground();
  }
  const Inst *body{bodies_.lookup(inst->symbol)};
  if (!body) {
    return Ground();
  }
  return Substitute(body, [&](const Inst &parameter) {
    return parameter.number < inst->arguments.size()
        ? inst->arguments[parameter.number]
        : &parameter;
  });
}

const Inst *InstContext::Substitute(const Inst *inst,
    llvm::function_ref<const Inst *(const Inst &)> replace) {
  llvm::DenseMap<const Inst *, const Inst *> done;
  std::function<const Inst *(const Inst *)> walk{[&](const Inst *i) {
    if (i->kind == InstKind::Parameter) {
      return replace(*i);
    }
    if (i->kind == InstKind::Constrained) {
      return Constrained(i->arguments, walk(i->constraint));
    }
    // A generated inst's body may refer to itself; it has no parameters
    // of its own to replace.
    if (i->kind == InstKind::Defined && !i->symbol) {
      return i;
    }
    if (auto it{done.find(i)}; it != done.end()) {
      return it->second;
    }
    Inst copy{*i};
    std::vector<std::vector<const Inst *>> arguments;
    std::vector<BoundAlternative> functors;
    for (const BoundAlternative &alternative : i->functors) {
      arguments.emplace_back();
      for (const Inst *argument : alternative.arguments) {
        arguments.back().push_back(walk(argument));
      }
    }
    for (std::size_t j{0}; j < i->functors.size(); ++j) {
      functors.push_back({i->functors[j].functor, arguments[j]});
    }
    copy.functors = functors;
    std::vector<ArgMode> modes;
    for (const ArgMode &mode : i->modes) {
      modes.push_back({walk(mode.initial), walk(mode.final)});
    }
    copy.modes = modes;
    std::vector<const Inst *> instArguments;
    for (const Inst *argument : i->arguments) {
      instArguments.push_back(walk(argument));
    }
    copy.arguments = instArguments;
    const Inst *result{Intern(copy)};
    done[i] = result;
    return result;
  }};
  return walk(inst);
}

const Inst *InstContext::Share(const Inst *inst, Uniqueness u) {
  llvm::DenseMap<const Inst *, const Inst *> done;
  // The generated insts that stand for named insts being shared, which
  // their bodies refer to if they are recursive.
  llvm::DenseSet<const Inst *> pending, used;
  // Whether sharing changes an inst, looking into named insts' bodies.
  llvm::DenseSet<const Inst *> seen;
  std::function<bool(const Inst *)> changes{[&](const Inst *i) {
    if (!seen.insert(i).second || i->kind == InstKind::Free ||
        i->kind == InstKind::Parameter) {
      return false;
    }
    if (i->kind == InstKind::Constrained) {
      return changes(i->constraint);
    }
    if (i->kind == InstKind::Defined) {
      return changes(Expand(i));
    }
    if (Least(i->uniqueness, u) != i->uniqueness) {
      return true;
    }
    return llvm::any_of(i->functors, [&](const BoundAlternative &alternative) {
      return llvm::any_of(alternative.arguments, changes);
    });
  }};
  std::function<const Inst *(const Inst *)> walk{[&](const Inst *i) {
    if (i->kind == InstKind::Free || i->kind == InstKind::Parameter ||
        (i->kind == InstKind::Defined && !i->symbol)) {
      return i;
    }
    if (i->kind == InstKind::Constrained) {
      return Constrained(i->arguments, walk(i->constraint));
    }
    if (auto it{done.find(i)}; it != done.end()) {
      if (pending.count(it->second)) {
        used.insert(it->second);
      }
      return it->second;
    }
    if (i->kind == InstKind::Defined) {
      // A named inst is shared as its body is, as in mmc, where
      // unique(...) in it is shared as well; one that sharing does not
      // change keeps its name.
      seen.clear();
      if (!changes(i)) {
        done[i] = i;
        return i;
      }
      const Inst *generated{Generated()};
      done[i] = generated;
      pending.insert(generated);
      const Inst *shared{walk(Expand(i))};
      pending.erase(generated);
      if (!used.count(generated)) {
        done[i] = shared;
        return shared;
      }
      // Unfolded once, so that it reads as the named inst does.
      Define(generated, shared);
      done[i] = shared;
      return shared;
    }
    Inst copy{*i};
    copy.uniqueness = Least(i->uniqueness, u);
    std::vector<std::vector<const Inst *>> arguments;
    std::vector<BoundAlternative> functors;
    for (const BoundAlternative &alternative : i->functors) {
      arguments.emplace_back();
      for (const Inst *argument : alternative.arguments) {
        arguments.back().push_back(walk(argument));
      }
    }
    for (std::size_t j{0}; j < i->functors.size(); ++j) {
      functors.push_back({i->functors[j].functor, arguments[j]});
    }
    copy.functors = functors;
    const Inst *result{Intern(copy)};
    done[i] = result;
    return result;
  }};
  return walk(inst);
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

namespace {
const char *GroundName(Uniqueness u) {
  switch (u) {
  case Uniqueness::Unique:
    return "unique";
  case Uniqueness::MostlyUnique:
    return "mostly_unique";
  case Uniqueness::Shared:
    return "ground";
  case Uniqueness::MostlyClobbered:
    return "mostly_clobbered";
  case Uniqueness::Clobbered:
    return "clobbered";
  }
  return "ground";
}

const char *AnyName(Uniqueness u) {
  switch (u) {
  case Uniqueness::Unique:
    return "unique_any";
  case Uniqueness::MostlyUnique:
    return "mostly_unique_any";
  case Uniqueness::Shared:
    return "any";
  case Uniqueness::MostlyClobbered:
    return "mostly_clobbered_any";
  case Uniqueness::Clobbered:
    return "clobbered_any";
  }
  return "any";
}

// The standard name of a mode, as the library defines it: in, out, di...
const char *StandardMode(const ArgMode &mode) {
  const Inst &i{*mode.initial}, &f{*mode.final};
  auto ground{[](const Inst &x, Uniqueness u) {
    return x.kind == InstKind::Ground && x.uniqueness == u;
  }};
  auto any{[](const Inst &x) {
    return x.kind == InstKind::Any && x.uniqueness == Uniqueness::Shared;
  }};
  bool free{i.kind == InstKind::Free};
  if (ground(i, Uniqueness::Shared) && ground(f, Uniqueness::Shared)) {
    return "in";
  }
  if (free && ground(f, Uniqueness::Shared)) {
    return "out";
  }
  if (free && ground(f, Uniqueness::Unique)) {
    return "uo";
  }
  if (ground(i, Uniqueness::Unique) && ground(f, Uniqueness::Unique)) {
    return "ui";
  }
  if (ground(i, Uniqueness::Unique) && ground(f, Uniqueness::Clobbered)) {
    return "di";
  }
  if (free && ground(f, Uniqueness::MostlyUnique)) {
    return "muo";
  }
  if (ground(i, Uniqueness::MostlyUnique) &&
      ground(f, Uniqueness::MostlyUnique)) {
    return "mui";
  }
  if (ground(i, Uniqueness::MostlyUnique) &&
      ground(f, Uniqueness::MostlyClobbered)) {
    return "mdi";
  }
  if (any(i) && any(f)) {
    return "ia";
  }
  if (free && any(f)) {
    return "oa";
  }
  if (free && f.kind == InstKind::Free) {
    return "unused";
  }
  return nullptr;
}

// An inst shares its parts, so as a tree it can be exponentially larger
// than it is; one too large to read is cut short after `maxParts` parts,
// as a type is (TypeToString).
class InstPrinter {
public:
  explicit InstPrinter(llvm::raw_ostream &os) : os_{os} {}

  void Print(const Inst &inst) {
    constexpr std::size_t maxParts{1000};
    if (++parts_ > maxParts) {
      if (parts_ == maxParts + 1) {
        os_ << "...";
      }
      return;
    }
    switch (inst.kind) {
    case InstKind::Free:
      os_ << "free";
      return;
    case InstKind::Any:
      os_ << AnyName(inst.uniqueness);
      return;
    case InstKind::Ground:
      os_ << GroundName(inst.uniqueness);
      return;
    case InstKind::Bound:
      if (inst.functors.empty()) {
        os_ << "not_reached";
        return;
      }
      os_ << (inst.uniqueness == Uniqueness::Shared
                 ? "bound"
                 : GroundName(inst.uniqueness))
          << '(';
      llvm::interleave(
          inst.functors, os_,
          [&](const BoundAlternative &alternative) {
            // Unqualified, as in mmc.
            Name(alternative.functor);
            if (!alternative.arguments.empty()) {
              os_ << '(';
              List(alternative.arguments);
              os_ << ')';
            }
          },
          " ; ");
      os_ << ')';
      return;
    case InstKind::HigherOrder: {
      if (inst.uniqueness == Uniqueness::Unique) {
        os_ << "/* unique */ ";
      }
      bool function{inst.predOrFunc == parser::PredOrFunc::Function};
      llvm::ArrayRef<ArgMode> arguments{inst.modes};
      if (function && !arguments.empty()) {
        arguments = arguments.drop_back();
      }
      os_ << '(' << (inst.any ? "any_" : "") << (function ? "func" : "pred");
      if (!arguments.empty()) {
        os_ << '(';
        llvm::interleave(
            arguments, os_, [&](const ArgMode &mode) { Mode(mode); }, ", ");
        os_ << ')';
      }
      if (function && !inst.modes.empty()) {
        os_ << " = ";
        Mode(inst.modes.back());
      }
      os_ << " is " << DeterminismName(inst.determinism) << ')';
      return;
    }
    case InstKind::Parameter:
      // mmc: (I =< ground), for an inst variable of a mode declaration.
      if (inst.constraint) {
        os_ << '(';
      }
      if (inst.name.empty()) {
        os_ << 'I' << inst.number;
      } else {
        os_ << inst.name;
      }
      if (inst.constraint) {
        os_ << " =< ";
        Print(*inst.constraint);
        os_ << ')';
      }
      return;
    case InstKind::Constrained:
      // mmc: (I, J =< bound(a)).
      os_ << '(';
      llvm::interleave(
          inst.arguments, os_,
          [&](const Inst *parameter) { os_ << parameter->name; }, ", ");
      os_ << " =< ";
      Print(*inst.constraint);
      os_ << ')';
      return;
    case InstKind::Defined:
      if (!inst.symbol) {
        os_ << "<inst " << inst.number << '>';
        return;
      }
      if (!inst.symbol->module.empty()) {
        os_ << inst.symbol->module << '.';
      }
      os_ << inst.symbol->name;
      if (!inst.arguments.empty()) {
        os_ << '(';
        List(inst.arguments);
        os_ << ')';
      }
      return;
    }
  }

  void Mode(const ArgMode &mode) {
    if (const char *name{StandardMode(mode)}) {
      os_ << name;
    } else if (mode.initial == mode.final) {
      os_ << "in(";
      Print(*mode.initial);
      os_ << ')';
    } else if (mode.initial->kind == InstKind::Free) {
      os_ << "out(";
      Print(*mode.final);
      os_ << ')';
    } else if (mode.final->kind == InstKind::Ground &&
        mode.final->uniqueness == Uniqueness::Clobbered) {
      // di(I) is I >> clobbered.
      os_ << "di(";
      Print(*mode.initial);
      os_ << ')';
    } else {
      os_ << '(';
      Print(*mode.initial);
      os_ << " >> ";
      Print(*mode.final);
      os_ << ')';
    }
  }

private:
  // A name that is not a word is quoted, as in mmc: '[|]', but [] and
  // {} are not, nor literals.
  void Name(const InstFunctor &functor) {
    llvm::StringRef name{functor.name};
    bool word{!name.empty() && llvm::isLower(name.front()) &&
        llvm::all_of(name, [](char c) { return llvm::isAlnum(c) || c == '_'; })};
    if (functor.literal || word || name == "[]" || name == "{}") {
      os_ << name;
    } else {
      os_ << '\'' << name << '\'';
    }
  }

  void List(llvm::ArrayRef<const Inst *> insts) {
    llvm::interleave(
        insts, os_, [&](const Inst *i) { Print(*i); }, ", ");
  }

  llvm::raw_ostream &os_;
  std::size_t parts_{0};
};
} // namespace

const char *DeterminismName(parser::Determinism determinism) {
  switch (determinism) {
  case parser::Determinism::Det:
    return "det";
  case parser::Determinism::Semidet:
    return "semidet";
  case parser::Determinism::Multi:
    return "multi";
  case parser::Determinism::Nondet:
    return "nondet";
  case parser::Determinism::Failure:
    return "failure";
  case parser::Determinism::Erroneous:
    return "erroneous";
  case parser::Determinism::CcMulti:
    return "cc_multi";
  case parser::Determinism::CcNondet:
    return "cc_nondet";
  }
  return "det";
}

void PrintInst(llvm::raw_ostream &os, const Inst &inst) {
  InstPrinter{os}.Print(inst);
}

std::string InstToString(const Inst &inst) {
  std::string text;
  llvm::raw_string_ostream os{text};
  PrintInst(os, inst);
  return text;
}

void PrintArgMode(llvm::raw_ostream &os, const ArgMode &mode) {
  InstPrinter{os}.Mode(mode);
}

std::string ArgModeToString(const ArgMode &mode) {
  std::string text;
  llvm::raw_string_ostream os{text};
  PrintArgMode(os, mode);
  return text;
}

//===----------------------------------------------------------------------===//
// InstOps
//===----------------------------------------------------------------------===//

InstTypes::~InstTypes() = default;

const std::vector<ConstructorTypes> *InstOps::ConstructorsOf(const Type *type) {
  auto [it, inserted]{constructors_.try_emplace(type)};
  if (inserted) {
    std::vector<ConstructorTypes> constructors;
    if (type && types_.Constructors(type, constructors)) {
      it->second = std::move(constructors);
    }
  }
  return it->second ? &*it->second : nullptr;
}

std::vector<const Type *> InstOps::ArgumentTypes(
    const InstFunctor &functor, const Type *type) {
  if (const std::vector<ConstructorTypes> *constructors{ConstructorsOf(type)}) {
    for (const ConstructorTypes &constructor : *constructors) {
      if (SameFunctor(constructor.functor, functor)) {
        return constructor.arguments.vec();
      }
    }
  }
  return std::vector<const Type *>(functor.arity, nullptr);
}

const Inst *InstOps::AsBound(const Inst *inst, const Type *type) {
  const std::vector<ConstructorTypes> *constructors{ConstructorsOf(type)};
  if (!constructors) {
    return nullptr;
  }
  std::vector<BoundAlternative> alternatives;
  std::vector<std::vector<const Inst *>> arguments;
  for (const ConstructorTypes &constructor : *constructors) {
    arguments.emplace_back(constructor.functor.arity, inst);
  }
  for (std::size_t i{0}; i < constructors->size(); ++i) {
    alternatives.push_back({(*constructors)[i].functor, arguments[i]});
  }
  return Bound(inst->uniqueness, alternatives);
}

const Inst *InstOps::Bound(
    Uniqueness u, llvm::ArrayRef<BoundAlternative> alternatives) {
  return insts_.Bound(u, alternatives);
}

const Inst *InstOps::Resolve(const Inst *inst) {
  // An inst definition that expands only to itself was reported, and
  // expands to ground.
  for (unsigned steps{0}; inst->kind == InstKind::Defined; ++steps) {
    if (steps == 100) {
      return insts_.Ground();
    }
    inst = insts_.Expand(inst);
  }
  return inst;
}

std::vector<ArgMode> InstOps::Instantiate(llvm::ArrayRef<ArgMode> modes,
    llvm::ArrayRef<const Inst *> actual, llvm::ArrayRef<const Type *> types,
    llvm::ArrayRef<const Inst *> fixed) {
  std::map<const Inst *, const Inst *> bound;
  for (std::size_t i{0}; i < modes.size() && i < actual.size(); ++i) {
    std::map<const Inst *, const Inst *> here;
    std::set<std::pair<const Inst *, const Inst *>> seen;
    Bind(actual[i], modes[i].initial, types[i], fixed, here, seen);
    for (const auto &[parameter, inst] : here) {
      bound.emplace(parameter, inst); // an earlier argument's stays
    }
  }
  if (bound.empty()) {
    return modes.vec();
  }
  auto replace{[&](const Inst &parameter) {
    auto it{bound.find(&parameter)};
    return it == bound.end() ? &parameter : it->second;
  }};
  std::vector<ArgMode> result;
  for (const ArgMode &mode : modes) {
    result.push_back({insts_.Substitute(mode.initial, replace),
        insts_.Substitute(mode.final, replace)});
  }
  return result;
}

void InstOps::Bind(const Inst *actual, const Inst *initial, const Type *type,
    llvm::ArrayRef<const Inst *> fixed,
    std::map<const Inst *, const Inst *> &bound,
    std::set<std::pair<const Inst *, const Inst *>> &seen) {
  if (!seen.insert({actual, initial}).second) {
    return;
  }
  if (initial->kind == InstKind::Parameter) {
    // Nor is one bound to an inst variable that a unification narrowed,
    // as (I =< bound(a)), which then does not match it, as in mmc (see the
    // design document's register).
    if (llvm::is_contained(fixed, initial) ||
        Resolve(actual)->kind == InstKind::Constrained) {
      return;
    }
    const Inst *constraint{
        initial->constraint ? initial->constraint : insts_.Ground()};
    if (!Matches(actual, constraint, type)) {
      return;
    }
    if (const Inst *inst{Unify(actual, constraint, type)}) {
      auto [it, added] {bound.emplace(initial, inst)};
      if (!added) {
        if (const Inst *merged{Merge(it->second, inst, type)}) {
          it->second = merged;
        }
      }
    }
    return;
  }
  const Inst *a{Plain(actual)};
  const Inst *b{Resolve(initial)};
  if (b->kind == InstKind::Bound) {
    if (a->kind == InstKind::Bound) {
      for (const BoundAlternative &alternative : a->functors) {
        if (const BoundAlternative *other{Find(b->functors, alternative.functor)}) {
          std::vector<const Type *> argumentTypes{
              ArgumentTypes(alternative.functor, type)};
          for (std::size_t i{0}; i < alternative.arguments.size(); ++i) {
            Bind(alternative.arguments[i], other->arguments[i],
                argumentTypes[i], fixed, bound, seen);
          }
        }
      }
    } else if (a->kind == InstKind::Ground || a->kind == InstKind::Any) {
      // Each argument of a ground term is ground.
      for (const BoundAlternative &alternative : b->functors) {
        std::vector<const Type *> argumentTypes{
            ArgumentTypes(alternative.functor, type)};
        for (std::size_t i{0}; i < alternative.arguments.size(); ++i) {
          Bind(a, alternative.arguments[i], argumentTypes[i], fixed, bound, seen);
        }
      }
    }
  } else if (b->kind == InstKind::HigherOrder &&
      a->kind == InstKind::HigherOrder && a->modes.size() == b->modes.size()) {
    for (std::size_t i{0}; i < a->modes.size(); ++i) {
      Bind(a->modes[i].initial, b->modes[i].initial, nullptr, fixed, bound, seen);
      Bind(a->modes[i].final, b->modes[i].final, nullptr, fixed, bound, seen);
    }
  }
}

const Inst *InstOps::Plain(const Inst *inst) {
  inst = Resolve(inst);
  if (inst->kind == InstKind::Parameter || inst->kind == InstKind::Constrained) {
    inst = Resolve(inst->constraint ? inst->constraint : insts_.Ground());
  }
  return inst;
}

std::pair<std::vector<const Inst *>, const Inst *> InstOps::Split(
    const Inst *inst) {
  switch (inst->kind) {
  case InstKind::Parameter:
    return {{inst}, inst->constraint ? inst->constraint : insts_.Ground()};
  case InstKind::Constrained:
    return {inst->arguments.vec(), inst->constraint};
  default:
    return {{}, inst};
  }
}

bool InstOps::SubInst(const Inst *a, const Inst *b, const Type *type) {
  return Matches(a, b, type) && Instantiated(a, b, type);
}

bool InstOps::IsFree(const Inst *inst) {
  return Resolve(inst)->kind == InstKind::Free;
}

bool InstOps::IsNotReached(const Inst *inst) {
  const Inst *resolved{Plain(inst)};
  return resolved->kind == InstKind::Bound && resolved->functors.empty();
}

std::optional<std::vector<const Inst *>> InstOps::Arguments(
    const Inst *inst, const InstFunctor &functor, const Type *type) {
  const Inst *resolved{Plain(inst)};
  switch (resolved->kind) {
  case InstKind::Free:
  case InstKind::Any:
  case InstKind::Ground:
    return std::vector<const Inst *>(functor.arity,
        resolved->kind == InstKind::Free ? resolved
            : resolved->kind == InstKind::Any
            ? insts_.Any(resolved->uniqueness)
            : insts_.Ground(resolved->uniqueness));
  case InstKind::Bound:
    if (const BoundAlternative *alternative{
            Find(resolved->functors, functor)}) {
      return alternative->arguments.vec();
    }
    return std::nullopt;
  default:
    return std::vector<const Inst *>(functor.arity, insts_.Ground());
  }
}

bool InstOps::IsGround(const Inst *inst, const Type *type) {
  return Instantiated(inst, insts_.Ground(Uniqueness::Unique), type);
}

// The manual's traversal of the inst tree of X with its type tree and
// the result's. A recursive inst node, one that is its own ancestor, needs
// a type at most the result's. A bound node keeps its functors, which the
// result type must have; a ground one of the same type stays, and one of a
// subtype becomes the bound inst of the subtype's constructors, down to a
// recursive type node, as ground.
const Inst *InstOps::Coerce(const Inst *inst, const Type *from,
    const Type *to,
    llvm::function_ref<bool(const Type *, const Type *)> subtype,
    CoerceFailure &failure) {
  auto fail{[&](const Inst *i, const Type *x, const Type *y)
                -> const Inst * {
    failure.from = x;
    failure.to = y;
    failure.inst = i;
    return nullptr;
  }};
  std::vector<const Type *> types;
  std::function<const Inst *(const Type *, const Type *, Uniqueness)> fromGround{
      [&](const Type *x, const Type *y, Uniqueness u) -> const Inst * {
        const std::vector<ConstructorTypes> *constructors{ConstructorsOf(x)};
        if (x == y || !constructors || llvm::is_contained(types, x)) {
          return insts_.Ground(u);
        }
        types.push_back(x);
        std::vector<std::vector<const Inst *>> arguments;
        for (const ConstructorTypes &constructor : *constructors) {
          std::vector<const Type *> ys{ArgumentTypes(constructor.functor, y)};
          std::vector<const Inst *> &args{arguments.emplace_back()};
          for (std::size_t j{0}; j < constructor.arguments.size(); ++j) {
            args.push_back(fromGround(constructor.arguments[j], ys[j], u));
          }
        }
        types.pop_back();
        std::vector<BoundAlternative> alternatives;
        for (std::size_t k{0}; k < constructors->size(); ++k) {
          alternatives.push_back({(*constructors)[k].functor, arguments[k]});
        }
        return Bound(u, alternatives);
      }};
  std::vector<const Inst *> ancestors;
  std::function<const Inst *(const Inst *, const Type *, const Type *)> walk{
      [&](const Inst *i, const Type *x, const Type *y) -> const Inst * {
        if (llvm::is_contained(ancestors, i)) {
          return x == y || subtype(x, y) ? i : fail(i, x, y);
        }
        const Inst *node{i};
        for (const Inst *next{Resolve(Plain(node))}; next != node;
             next = Resolve(Plain(node))) {
          node = next;
        }
        switch (node->kind) {
        case InstKind::Bound: {
          if (node->functors.empty()) {
            return node; // not reached
          }
          ancestors.push_back(i);
          std::vector<std::vector<const Inst *>> arguments;
          for (const BoundAlternative &alternative : node->functors) {
            const std::vector<ConstructorTypes> *constructors{
                ConstructorsOf(y)};
            if (x != y &&
                (!constructors ||
                    llvm::none_of(*constructors, [&](const ConstructorTypes &c) {
                      return SameFunctor(c.functor, alternative.functor);
                    }))) {
              failure.functor = alternative.functor;
              return fail(node, x, y);
            }
            std::vector<const Type *> xs{ArgumentTypes(alternative.functor, x)};
            std::vector<const Type *> ys{ArgumentTypes(alternative.functor, y)};
            std::vector<const Inst *> &args{arguments.emplace_back()};
            for (std::size_t j{0}; j < alternative.arguments.size(); ++j) {
              failure.path.push_back({alternative.functor, j});
              const Inst *arg{walk(alternative.arguments[j], xs[j], ys[j])};
              if (!arg) {
                return nullptr;
              }
              failure.path.pop_back();
              args.push_back(arg);
            }
          }
          ancestors.pop_back();
          std::vector<BoundAlternative> alternatives;
          for (std::size_t k{0}; k < node->functors.size(); ++k) {
            alternatives.push_back({node->functors[k].functor, arguments[k]});
          }
          return Bound(node->uniqueness, alternatives);
        }
        case InstKind::Ground:
          if (x == y) {
            return i;
          }
          return subtype(x, y) ? fromGround(x, y, node->uniqueness)
                               : fail(node, x, y);
        case InstKind::Free:
          return fail(node, x, y);
        default:
          return x == y ? i : fail(node, x, y);
        }
      }};
  return walk(inst, from, to);
}

const Inst *InstOps::Normalise(const Inst *inst, const Type *type) {
  if (IsNotReached(inst) || !IsGround(inst, type) ||
      Resolve(inst)->kind == InstKind::Constrained ||
      Resolve(inst)->kind == InstKind::Parameter) {
    return inst;
  }
  Uniqueness u{Uniqueness::Unique};
  llvm::DenseSet<const Inst *> seen;
  std::function<void(const Inst *)> walk{[&](const Inst *i) {
    if (!seen.insert(i).second) {
      return;
    }
    i = Resolve(Plain(i));
    if (i->kind != InstKind::Free) {
      u = std::max(u, i->uniqueness);
    }
    for (const BoundAlternative &alternative : i->functors) {
      for (const Inst *argument : alternative.arguments) {
        walk(argument);
      }
    }
  }};
  walk(inst);
  return insts_.Ground(u);
}

bool InstOps::HasFree(const Inst *inst, const Type *type) {
  return Contains(inst, type, InstKind::Free);
}

bool InstOps::HasAny(const Inst *inst, const Type *type) {
  return Contains(inst, type, InstKind::Any);
}

bool InstOps::Contains(const Inst *inst, const Type *type, InstKind kind) {
  llvm::DenseSet<std::pair<const Inst *, const Type *>> seen;
  std::function<bool(const Inst *, const Type *)> walk{
      [&](const Inst *i, const Type *t) {
        if (!seen.insert({i, t}).second) {
          return false;
        }
        i = Plain(i);
        if (i->kind == kind) {
          return true;
        }
        for (const BoundAlternative &alternative : i->functors) {
          std::vector<const Type *> types{ArgumentTypes(alternative.functor, t)};
          for (std::size_t j{0}; j < alternative.arguments.size(); ++j) {
            if (walk(alternative.arguments[j], types[j])) {
              return true;
            }
          }
        }
        return false;
      }};
  return walk(inst, type);
}

bool InstOps::Matches(const Inst *a, const Inst *b, const Type *type) {
  if (a == b) {
    return true;
  }
  if (a->kind != InstKind::Defined && b->kind != InstKind::Defined) {
    return MatchesNow(a, b, type);
  }
  // Recursive insts match if no difference is found below the pair
  // already being compared.
  Key key{a, b, type};
  if (matching_.count(key)) {
    return true;
  }
  matching_[key] = true;
  bool matches{MatchesNow(Resolve(a), Resolve(b), type)};
  matching_.erase(key);
  return matches;
}

bool InstOps::MatchesNow(const Inst *a, const Inst *b, const Type *type) {
  if (a == b) {
    return true;
  }
  if (a->kind == InstKind::Bound && a->functors.empty()) {
    return true; // not_reached matches every inst
  }
  if (a->kind == InstKind::Parameter || a->kind == InstKind::Constrained ||
      b->kind == InstKind::Parameter || b->kind == InstKind::Constrained) {
    // To match an inst with inst variables, an inst must be known to be
    // compatible with what each of them stands for (definition 4.2.3).
    auto [as, ai] {Split(a)};
    auto [bs, bi] {Split(b)};
    return llvm::all_of(bs, [&](const Inst *p) { return llvm::is_contained(as, p); }) &&
        Matches(ai, bi, type);
  }
  if (b->kind == InstKind::Free || a->kind == InstKind::Free) {
    return a->kind == b->kind;
  }
  if (!AtLeastAsUnique(a->uniqueness, b->uniqueness)) {
    return false;
  }
  auto arguments{[&](const BoundAlternative &alternative, const Inst *with) {
    std::vector<const Type *> types{ArgumentTypes(alternative.functor, type)};
    for (std::size_t i{0}; i < alternative.arguments.size(); ++i) {
      if (!Matches(alternative.arguments[i], with, types[i])) {
        return false;
      }
    }
    return true;
  }};
  switch (b->kind) {
  case InstKind::Any:
  case InstKind::Ground: {
    if (a->kind == InstKind::Any) {
      return b->kind == InstKind::Any;
    }
    if (a->kind == InstKind::HigherOrder) {
      return a->predOrFunc == parser::PredOrFunc::Predicate ||
          IsDefaultFunction(a);
    }
    if (a->kind != InstKind::Bound) {
      return true; // ground
    }
    const Inst *with{b->kind == InstKind::Any ? insts_.Any(b->uniqueness)
                                              : insts_.Ground(b->uniqueness)};
    return llvm::all_of(a->functors, [&](const BoundAlternative &alternative) {
      return arguments(alternative, with);
    });
  }
  case InstKind::Bound: {
    if (a->kind == InstKind::Ground) {
      const Inst *bound{AsBound(a, type)};
      return bound && MatchesNow(bound, b, type);
    }
    if (a->kind != InstKind::Bound) {
      return false;
    }
    for (const BoundAlternative &alternative : a->functors) {
      const BoundAlternative *other{Find(b->functors, alternative.functor)};
      if (!other) {
        return false;
      }
      std::vector<const Type *> types{ArgumentTypes(alternative.functor, type)};
      for (std::size_t i{0}; i < alternative.arguments.size(); ++i) {
        if (!Matches(alternative.arguments[i], other->arguments[i], types[i])) {
          return false;
        }
      }
    }
    return true;
  }
  case InstKind::HigherOrder:
    if (a->kind == InstKind::Ground) {
      return IsDefaultFunction(b);
    }
    return a->kind == InstKind::HigherOrder &&
        a->predOrFunc == b->predOrFunc && a->any == b->any &&
        a->modes == b->modes && a->determinism == b->determinism;
  default:
    return false;
  }
}

const Inst *InstOps::DefaultFunction(std::size_t arity) {
  std::vector<ArgMode> modes(
      arity, ArgMode{insts_.Ground(), insts_.Ground()});
  modes.push_back(ArgMode{insts_.Free(), insts_.Ground()});
  return insts_.HigherOrder(parser::PredOrFunc::Function, false, modes,
      parser::Determinism::Det);
}

bool InstOps::IsDefaultFunction(const Inst *inst) {
  if (inst->kind != InstKind::HigherOrder ||
      inst->predOrFunc != parser::PredOrFunc::Function || inst->any ||
      inst->modes.empty()) {
    return false;
  }
  const Inst *standard{DefaultFunction(inst->modes.size() - 1)};
  return inst->modes == standard->modes &&
      inst->determinism == standard->determinism;
}

bool InstOps::Instantiated(const Inst *a, const Inst *b, const Type *type) {
  if (a == b) {
    return true;
  }
  if (a->kind != InstKind::Defined && b->kind != InstKind::Defined) {
    return InstantiatedNow(a, b, type);
  }
  Key key{a, b, type};
  if (instantiating_.count(key)) {
    return true;
  }
  instantiating_[key] = true;
  bool instantiated{InstantiatedNow(Resolve(a), Resolve(b), type)};
  instantiating_.erase(key);
  return instantiated;
}

bool InstOps::InstantiatedNow(const Inst *a, const Inst *b, const Type *type) {
  if (a == b || b->kind == InstKind::Free) {
    return true;
  }
  if (a->kind == InstKind::Free) {
    return false;
  }
  if (a->kind == InstKind::Bound && a->functors.empty()) {
    return true;
  }
  if (a->kind == InstKind::Parameter || a->kind == InstKind::Constrained ||
      b->kind == InstKind::Parameter || b->kind == InstKind::Constrained) {
    // A variable keeps the inst variables it gets only while its inst
    // stays a sub-inst of the one it had (definition 4.2.2).
    auto [as, ai] {Split(a)};
    auto [bs, bi] {Split(b)};
    if (!Instantiated(ai, bi, type)) {
      return false;
    }
    auto in{[&](const Inst *p) { return llvm::is_contained(as, p); }};
    return SubInst(ai, bi, type) ? llvm::all_of(bs, in) : llvm::none_of(bs, in);
  }
  // A variable never becomes more unique on forward execution.
  if (!AtLeastAsUnique(b->uniqueness, a->uniqueness)) {
    return false;
  }
  switch (b->kind) {
  case InstKind::Any:
  case InstKind::Ground: {
    if (a->kind == InstKind::Any) {
      return b->kind == InstKind::Any;
    }
    if (a->kind != InstKind::Bound) {
      return true;
    }
    const Inst *with{b->kind == InstKind::Any ? insts_.Any(b->uniqueness)
                                              : insts_.Ground(b->uniqueness)};
    for (const BoundAlternative &alternative : a->functors) {
      std::vector<const Type *> types{ArgumentTypes(alternative.functor, type)};
      for (std::size_t i{0}; i < alternative.arguments.size(); ++i) {
        if (!Instantiated(alternative.arguments[i], with, types[i])) {
          return false;
        }
      }
    }
    return true;
  }
  case InstKind::Bound: {
    if (a->kind == InstKind::Ground) {
      const Inst *bound{AsBound(a, type)};
      return bound && InstantiatedNow(bound, b, type);
    }
    if (a->kind != InstKind::Bound) {
      return false;
    }
    for (const BoundAlternative &alternative : a->functors) {
      const BoundAlternative *other{Find(b->functors, alternative.functor)};
      if (!other) {
        return false;
      }
      std::vector<const Type *> types{ArgumentTypes(alternative.functor, type)};
      for (std::size_t i{0}; i < alternative.arguments.size(); ++i) {
        if (!Instantiated(
                alternative.arguments[i], other->arguments[i], types[i])) {
          return false;
        }
      }
    }
    return true;
  }
  case InstKind::HigherOrder:
    return a->kind == InstKind::HigherOrder &&
        a->predOrFunc == b->predOrFunc && a->any == b->any &&
        a->modes == b->modes && a->determinism == b->determinism;
  default:
    return false;
  }
}

const Inst *InstOps::Merge(const Inst *a, const Inst *b, const Type *type) {
  if (a == b) {
    return a;
  }
  if (a->kind != InstKind::Defined && b->kind != InstKind::Defined) {
    return MergeNow(a, b, type);
  }
  // A merge that reaches itself again refers to a Generated inst, which
  // becomes the result.
  Key key{a, b, type};
  if (auto it{merged_.find(key)}; it != merged_.end()) {
    return it->second;
  }
  if (auto it{merging_.find(key)}; it != merging_.end()) {
    if (!it->second) {
      it->second = insts_.Generated();
    }
    return it->second;
  }
  merging_[key] = nullptr;
  const Inst *result{MergeNow(Resolve(a), Resolve(b), type)};
  const Inst *generated{merging_[key]};
  merging_.erase(key);
  if (generated && result) {
    insts_.Define(generated, result);
    result = generated;
  }
  merged_[key] = result;
  return result;
}

const Inst *InstOps::MergeNow(const Inst *a, const Inst *b, const Type *type) {
  if (a == b) {
    return a;
  }
  if (a->kind == InstKind::Bound && a->functors.empty()) {
    return b;
  }
  if (b->kind == InstKind::Bound && b->functors.empty()) {
    return a;
  }
  if (a->kind == InstKind::Parameter || a->kind == InstKind::Constrained ||
      b->kind == InstKind::Parameter || b->kind == InstKind::Constrained) {
    // The inst variables of both (section 4.2.3).
    auto [as, ai] {Split(a)};
    auto [bs, bi] {Split(b)};
    const Inst *merged{Merge(ai, bi, type)};
    if (!merged) {
      return nullptr;
    }
    std::vector<const Inst *> both;
    for (const Inst *p : as) {
      if (llvm::is_contained(bs, p)) {
        both.push_back(p);
      }
    }
    return insts_.Constrained(both, merged);
  }
  if (a->kind == InstKind::Free || b->kind == InstKind::Free) {
    return nullptr;
  }
  Uniqueness u{Least(a->uniqueness, b->uniqueness)};
  if (a->kind == InstKind::Any || b->kind == InstKind::Any) {
    const Inst *any{insts_.Any(u)};
    return Matches(a, any, type) && Matches(b, any, type) ? any : nullptr;
  }
  // Higher-order insts that differ merge to ground, which both match.
  auto plain{[&](const Inst *x) {
    return x->kind == InstKind::HigherOrder ? insts_.Ground(x->uniqueness) : x;
  }};
  if (a->kind == InstKind::HigherOrder || b->kind == InstKind::HigherOrder) {
    return Merge(plain(a), plain(b), type);
  }
  if (a->kind == InstKind::Ground || b->kind == InstKind::Ground) {
    const Inst *ground{insts_.Ground(u)};
    if (Matches(a, ground, type) && Matches(b, ground, type)) {
      return ground;
    }
    // A bound inst with free or any parts: the lub is a bound inst.
    const Inst *x{a->kind == InstKind::Ground ? AsBound(a, type) : a};
    const Inst *y{b->kind == InstKind::Ground ? AsBound(b, type) : b};
    return x && y ? MergeNow(x, y, type) : nullptr;
  }
  std::vector<BoundAlternative> alternatives{
      a->functors.begin(), a->functors.end()};
  std::vector<std::vector<const Inst *>> merged;
  merged.reserve(b->functors.size());
  for (const BoundAlternative &alternative : b->functors) {
    auto it{llvm::find_if(alternatives, [&](const BoundAlternative &x) {
      return SameFunctor(x.functor, alternative.functor);
    })};
    if (it == alternatives.end()) {
      alternatives.push_back(alternative);
      continue;
    }
    std::vector<const Type *> types{ArgumentTypes(alternative.functor, type)};
    merged.emplace_back();
    for (std::size_t i{0}; i < alternative.arguments.size(); ++i) {
      const Inst *argument{
          Merge(it->arguments[i], alternative.arguments[i], types[i])};
      if (!argument) {
        return nullptr;
      }
      merged.back().push_back(argument);
    }
    it->arguments = merged.back();
  }
  return Bound(u, alternatives);
}

const Inst *InstOps::Unify(const Inst *a, const Inst *b, const Type *type) {
  if (a == b) {
    return a;
  }
  if (a->kind != InstKind::Defined && b->kind != InstKind::Defined) {
    return UnifyNow(a, b, type);
  }
  Key key{a, b, type};
  if (auto it{unified_.find(key)}; it != unified_.end()) {
    return it->second;
  }
  if (auto it{unifying_.find(key)}; it != unifying_.end()) {
    if (!it->second) {
      it->second = insts_.Generated();
    }
    return it->second;
  }
  unifying_[key] = nullptr;
  const Inst *result{UnifyNow(Resolve(a), Resolve(b), type)};
  const Inst *generated{unifying_[key]};
  unifying_.erase(key);
  if (generated && result) {
    insts_.Define(generated, result);
    result = generated;
  }
  unified_[key] = result;
  return result;
}

const Inst *InstOps::UnifyNow(const Inst *a, const Inst *b, const Type *type) {
  if (a == b) {
    return a;
  }
  if (a->kind == InstKind::Free) {
    return b;
  }
  if (b->kind == InstKind::Free) {
    return a;
  }
  if ((a->kind == InstKind::Bound && a->functors.empty()) ||
      (b->kind == InstKind::Bound && b->functors.empty())) {
    return insts_.NotReached();
  }
  if (a->kind == InstKind::Parameter || a->kind == InstKind::Constrained ||
      b->kind == InstKind::Parameter || b->kind == InstKind::Constrained) {
    // The inst variables of each side whose inst the result is a sub-inst
    // of (section 4.2.3).
    auto [as, ai] {Split(a)};
    auto [bs, bi] {Split(b)};
    const Inst *unified{Unify(ai, bi, type)};
    if (!unified) {
      return nullptr;
    }
    std::vector<const Inst *> kept;
    if (SubInst(unified, ai, type)) {
      kept = as;
    }
    if (SubInst(unified, bi, type)) {
      kept.insert(kept.end(), bs.begin(), bs.end());
    }
    return insts_.Constrained(kept, unified);
  }
  Uniqueness u{Least(a->uniqueness, b->uniqueness)};
  if (a->kind == InstKind::HigherOrder || b->kind == InstKind::HigherOrder) {
    const Inst *higher{a->kind == InstKind::HigherOrder ? a : b};
    const Inst *other{higher == a ? b : a};
    if (other->kind != InstKind::Ground && other->kind != InstKind::Any) {
      return nullptr; // two higher-order values, or one and a functor
    }
    return insts_.HigherOrder(higher->predOrFunc, higher->any, higher->modes,
        higher->determinism, u);
  }
  if (a->kind == InstKind::Any && b->kind == InstKind::Any) {
    return insts_.Any(u);
  }
  if (a->kind != InstKind::Bound && b->kind != InstKind::Bound) {
    return insts_.Ground(u); // ground with ground or any
  }
  // Each alternative of a bound inst, with its arguments unified with
  // those of the other side; one with a not_reached argument is dropped.
  auto unifyAlternative{[&](const BoundAlternative &alternative,
                            llvm::function_ref<const Inst *(std::size_t)> other,
                            std::vector<BoundAlternative> &out,
                            std::vector<std::vector<const Inst *>> &storage)
                            -> bool {
    std::vector<const Type *> types{ArgumentTypes(alternative.functor, type)};
    std::vector<const Inst *> arguments;
    for (std::size_t i{0}; i < alternative.arguments.size(); ++i) {
      const Inst *argument{Unify(alternative.arguments[i], other(i), types[i])};
      if (!argument) {
        return false;
      }
      if (IsNotReached(argument)) {
        return true;
      }
      arguments.push_back(argument);
    }
    storage.push_back(std::move(arguments));
    out.push_back({alternative.functor, {}});
    return true;
  }};
  std::vector<BoundAlternative> alternatives;
  std::deque<std::vector<const Inst *>> storage;
  std::vector<std::vector<const Inst *>> pending;
  auto finish{[&] {
    for (std::size_t i{0}; i < alternatives.size(); ++i) {
      storage.push_back(std::move(pending[i]));
      alternatives[i].arguments = storage.back();
    }
    return Bound(u, alternatives);
  }};
  if (a->kind != InstKind::Bound || b->kind != InstKind::Bound) {
    const Inst *bound{a->kind == InstKind::Bound ? a : b};
    const Inst *other{bound == a ? b : a};
    // Ground or any: each argument becomes at least as instantiated.
    const Inst *with{other->kind == InstKind::Any ? insts_.Any(u)
                                                  : insts_.Ground(u)};
    for (const BoundAlternative &alternative : bound->functors) {
      if (!unifyAlternative(
              alternative, [&](std::size_t) { return with; }, alternatives,
              pending)) {
        return nullptr;
      }
    }
    return finish();
  }
  for (const BoundAlternative &alternative : a->functors) {
    const BoundAlternative *other{Find(b->functors, alternative.functor)};
    if (!other) {
      continue;
    }
    if (!unifyAlternative(
            alternative, [&](std::size_t i) { return other->arguments[i]; },
            alternatives, pending)) {
      return nullptr;
    }
  }
  return finish();
}

std::optional<llvm::StringRef> WrongBuiltinInstArity(
    const parser::SymName &name, unsigned arity) {
  if (!name.qualifiers.empty()) {
    return std::nullopt;
  }
  if (llvm::is_contained({"bound", "bound_unique"}, name.name)) {
    return arity == 1 ? std::nullopt : std::optional<llvm::StringRef>{"one"};
  }
  if (llvm::is_contained({"unique", "mostly_unique"}, name.name)) {
    return arity <= 1 ? std::nullopt
                      : std::optional<llvm::StringRef>{"zero or one"};
  }
  if (arity > 0 &&
      llvm::is_contained({"free", "ground", "clobbered", "mostly_clobbered",
                             "any", "unique_any", "mostly_unique_any",
                             "clobbered_any", "mostly_clobbered_any",
                             "not_reached"},
          name.name)) {
    return "zero";
  }
  return std::nullopt;
}

} // namespace merkurius::semantics
