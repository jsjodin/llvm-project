//===-- lib/Semantics/check-modes.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/check-modes.h"
#include "merkurius/Common/idioms.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <string>

namespace merkurius::semantics {

using namespace parser::literals;

ModeChecking::ModeChecking() = default;
ModeChecking::~ModeChecking() = default;

llvm::ArrayRef<const Procedure *> ModeChecking::ProceduresOf(
    const Symbol &symbol) const {
  auto it{bySymbol_.find(&symbol)};
  return it == bySymbol_.end() ? llvm::ArrayRef<const Procedure *>{}
                               : llvm::ArrayRef<const Procedure *>{it->second};
}

namespace {

// The reserved inst and mode names (reference manual, "Insts, modes, and
// mode definitions").
bool ReservedInst(llvm::StringRef name) {
  return llvm::is_contained(
      {"=<", "any", "bound", "bound_unique", "clobbered", "clobbered_any",
          "free", "ground", "is", "mostly_clobbered", "mostly_unique",
          "mostly_unique_any", "not_reached", "unique", "unique_any"},
      name);
}

bool ReservedMode(llvm::StringRef name) {
  return llvm::is_contained(
      {"=", ">>", "any_func", "any_pred", "func", "is", "pred"}, name);
}

std::string Text(const Symbol &symbol) {
  std::string text;
  llvm::raw_string_ostream os{text};
  os << symbol;
  return text;
}

// A procedure that is erroneous or failure never returns a solution, so
// nothing after a call of it is reached, as mode analysis knows (reference
// manual, "Replacing compile-time checking with run-time checking").
bool NoSolutions(parser::Determinism determinism) {
  return determinism == parser::Determinism::Erroneous ||
      determinism == parser::Determinism::Failure;
}

const char *Keyword(parser::PredOrFunc kind) {
  return kind == parser::PredOrFunc::Function ? "func" : "pred";
}

// The constructors of a type, from its definition, for the inst
// operations.
class ConstructorTable : public InstTypes {
public:
  ConstructorTable(const NameResolution &resolution, TypeChecking &types)
      : resolution_{resolution}, types_{types} {}

  bool Constructors(
      const Type *type, std::vector<ConstructorTypes> &out) override {
    if (!type) {
      return false;
    }
    if (type->kind == TypeKind::Tuple) {
      out.push_back({{{}, "{}", static_cast<unsigned>(type->arguments.size()),
                         false},
          type->arguments});
      return true;
    }
    if (type->kind != TypeKind::Named ||
        !std::holds_alternative<const parser::DuTypeDefn *>(
            type->constructor->declaration)) {
      return false;
    }
    const Symbol &symbol{*type->constructor};
    const ModuleSymbols *symbols{symbol.module == resolution_.module().name()
            ? &resolution_.module()
            : resolution_.Module(symbol.module)};
    if (!symbols) {
      return false;
    }
    for (const Symbol *constructor : symbols->symbols()) {
      if (constructor->kind != SymbolKind::Constructor ||
          constructor->owner != &symbol) {
        continue;
      }
      const Signature *signature{types_.SignatureOf(*constructor)};
      if (!signature) {
        return false;
      }
      std::vector<const Type *> arguments;
      for (const Type *argument : signature->arguments.drop_back()) {
        arguments.push_back(types_.types().Substitute(
            argument, [&](unsigned var) -> const Type * {
              return var < type->arguments.size() ? type->arguments[var]
                                                  : nullptr;
            }));
      }
      out.push_back({{constructor->module, constructor->name,
                         constructor->arity, false},
          arena_.SaveArray(llvm::ArrayRef<const Type *>{arguments})});
    }
    return true;
  }

private:
  const NameResolution &resolution_;
  TypeChecking &types_;
  common::Arena arena_;
};

} // namespace

class ModeChecker {
public:
  ModeChecker(ModeChecking &result, const NameResolution &resolution,
      TypeChecking &types, const parser::TokenSequence &tokens,
      parser::Messages &messages, const ModeCheckOptions &options)
      : result_{result}, insts_{result.insts_}, resolution_{resolution},
        types_{types}, tokens_{tokens}, messages_{messages}, options_{options},
        constructors_{resolution, types} {}

  void Run(const parser::ParseTree &);
  // A lambda expression's argument modes, a function's result last: an
  // argument without one is in, and a function's result out (reference
  // manual, "Creating higher-order terms"). None if they have inst
  // variables.
  std::optional<std::vector<ArgMode>> LambdaModes(const Lambda &);

private:
  // The inst variables of a definition or a declaration as they are
  // converted. A definition's must be its parameters; a declaration's are
  // its inst parameters, numbered as they occur, each with one constraint,
  // ground if none is given (reference manual, "Constrained polymorphic
  // modes").
  struct InstScope {
    bool definition{false};
    llvm::ArrayRef<llvm::StringRef> parameters{};
    llvm::StringRef freeParameter{}; // the first that is not one
    std::vector<llvm::StringRef> names{};
    std::vector<const Inst *> constraints{}; // explicit ones, or null
    std::vector<const Inst *> made{};
    llvm::StringRef inconsistent{};
    unsigned id{0}; // a declaration's, numbered from 1
  };

  bool Own(const Symbol &symbol) const {
    return symbol.module == resolution_.module().name();
  }
  parser::CharBlock Source(parser::TokenRange range) const {
    return {tokensNow_->tokens[range.first].spelling.begin(),
        tokensNow_->tokens[range.last].spelling.end()};
  }
  parser::CharBlock Source(const parser::SymName &name) const {
    return Source(name.tokens);
  }

  const Inst *Convert(const parser::Inst &, InstScope &);
  ArgMode Convert(const parser::Mode &, InstScope &);
  void Collect(const parser::Inst &, InstScope &);
  void Collect(const parser::Mode &, InstScope &);
  const Inst *Parameter(llvm::StringRef name, InstScope &);
  const Symbol *Named(const parser::SymName &, SymbolKind, unsigned arity);
  void DefineInst(const Symbol &);
  std::optional<ArgMode> DefineMode(const Symbol &);
  void CheckInstCycles();

  struct Declared {
    std::vector<ArgMode> modes;
    std::optional<parser::Determinism> determinism;
    std::vector<const Inst *> parameters;
  };
  Declared ConvertDeclaration(llvm::ArrayRef<parser::Mode> arguments,
      std::optional<parser::Mode> result,
      std::optional<parser::Inst> withInst,
      std::optional<parser::Determinism>,
      llvm::ArrayRef<const parser::InstConstraint *> constraints,
      parser::CharBlock where);
  const Symbol *ModeDeclTarget(const parser::ModeDecl &,
      const Symbol *typeclass, const ModuleSymbols &);
  void BuildProcedures(llvm::ArrayRef<parser::Item>, const ModuleSymbols &);
  void CheckForTypes(
      const Symbol &, llvm::ArrayRef<ArgMode>, parser::CharBlock where);

  ModeChecking &result_;
  InstContext &insts_;
  const NameResolution &resolution_;
  TypeChecking &types_;
  const parser::TokenSequence &tokens_;
  // Those of the items being read.
  const parser::TokenSequence *tokensNow_{&tokens_};
  parser::Messages &messages_;
  const ModeCheckOptions &options_;
  ConstructorTable constructors_;
  // Whether the item converted is the module's own, whose errors are
  // reported.
  bool own_{true};

  enum class State { InProgress, Done };
  llvm::DenseMap<const Symbol *, State> instState_, modeState_;
  llvm::DenseMap<const Symbol *, ArgMode> modes_;
  // Definitions with an inst parameter that is not a parameter: mmc does
  // not define them.
  llvm::DenseSet<const Symbol *> bad_;
  // The type constructor an inst definition is for.
  llvm::DenseMap<const Symbol *, const Symbol *> forTypes_;
  // The mode definitions being expanded, outermost first.
  std::vector<const Symbol *> expanding_;
  llvm::DenseSet<const Symbol *> cyclic_;
  // The declarations' inst variable scopes so far.
  unsigned scopes_{0};
};

//===----------------------------------------------------------------------===//
// Insts and modes
//===----------------------------------------------------------------------===//

const Symbol *ModeChecker::Named(
    const parser::SymName &name, SymbolKind kind, unsigned arity) {
  llvm::SmallVector<const Symbol *, 2> found;
  for (const Symbol *candidate : name.candidates) {
    if (candidate->kind == kind && candidate->arity == arity &&
        !llvm::is_contained(found, candidate)) {
      found.push_back(candidate);
    }
  }
  if (found.empty()) {
    return nullptr; // undefined, and reported
  }
  if (found.size() > 1 && own_) {
    // mmc: "ambiguity error: multiple possible matches for inst
    // `list_skel'/1".
    std::string list;
    llvm::interleave(
        found,
        [&](const Symbol *s) { list += s->module.str() + '.' + s->name.str(); },
        [&] { list += ", "; });
    messages_.Say(Source(name), "the %s name %s/%u is ambiguous: it may be %s"_err_en_US,
        kind == SymbolKind::Inst ? "inst" : "mode", name.name.str(), arity,
        list);
  }
  return found.front();
}

const Inst *ModeChecker::Parameter(llvm::StringRef name, InstScope &scope) {
  if (scope.definition) {
    const auto *it{llvm::find(scope.parameters, name)};
    if (it == scope.parameters.end()) {
      if (scope.freeParameter.empty()) {
        scope.freeParameter = name;
      }
      return insts_.Ground();
    }
    return insts_.Parameter(it - scope.parameters.begin(), name, nullptr);
  }
  auto it{llvm::find(scope.names, name)};
  std::size_t number{static_cast<std::size_t>(it - scope.names.begin())};
  if (it == scope.names.end()) {
    scope.names.push_back(name);
    scope.constraints.push_back(nullptr);
  }
  // The variables collected before it may not be converted yet, as `A`
  // in `mode ok(in(A), out(I)) <= I =< ground`, whose constraints name
  // `I` first.
  while (number >= scope.made.size()) {
    std::size_t next{scope.made.size()};
    // Unconstrained means compatible with ground.
    const Inst *constraint{scope.constraints[next]};
    scope.made.push_back(insts_.Parameter(next, scope.names[next],
        constraint ? constraint : insts_.Ground(), scope.id));
  }
  return scope.made[number];
}

// The constraints a declaration gives its inst variables, before any is
// converted, so that each occurrence has its constraint.
void ModeChecker::Collect(const parser::Inst &inst, InstScope &scope) {
  common::visit(
      common::visitors{
          [&](const parser::InstVar *var) {
            auto it{llvm::find(scope.names, var->name)};
            std::size_t number{
                static_cast<std::size_t>(it - scope.names.begin())};
            if (it == scope.names.end()) {
              scope.names.push_back(var->name);
              scope.constraints.push_back(nullptr);
            }
            if (var->constraint) {
              InstScope inner;
              const Inst *constraint{Convert(*var->constraint, inner)};
              if (!scope.constraints[number]) {
                scope.constraints[number] = constraint;
              } else if (scope.constraints[number] != constraint &&
                  scope.inconsistent.empty()) {
                scope.inconsistent = var->name;
              }
            }
          },
          [&](const parser::BoundInst *bound) {
            for (const parser::BoundFunctor *functor : bound->functors) {
              for (const parser::Inst &argument : functor->arguments) {
                Collect(argument, scope);
              }
            }
          },
          [&](const parser::HigherOrderInst *higher) {
            for (const parser::Mode &mode : higher->arguments) {
              Collect(mode, scope);
            }
            if (higher->result) {
              Collect(*higher->result, scope);
            }
          },
          [&](const parser::NamedInst *named) {
            for (const parser::Inst &argument : named->arguments) {
              Collect(argument, scope);
            }
          },
          [](const parser::BuiltinInst *) {},
      },
      inst);
}

void ModeChecker::Collect(const parser::Mode &mode, InstScope &scope) {
  common::visit(
      common::visitors{
          [&](const parser::MappingMode *mapping) {
            Collect(mapping->initial, scope);
            Collect(mapping->final, scope);
          },
          [&](const parser::NamedMode *named) {
            for (const parser::Inst &argument : named->arguments) {
              Collect(argument, scope);
            }
          },
          [&](const parser::HigherOrderInst *higher) {
            Collect(parser::Inst{higher}, scope);
          },
      },
      mode);
}

const Inst *ModeChecker::Convert(const parser::Inst &inst, InstScope &scope) {
  return common::visit(
      common::visitors{
          [&](const parser::BuiltinInst *builtin) -> const Inst * {
            using K = parser::BuiltinInstKind;
            switch (builtin->kind) {
            case K::Free:
              return insts_.Free();
            case K::Ground:
              return insts_.Ground();
            case K::Unique:
              return insts_.Ground(Uniqueness::Unique);
            case K::MostlyUnique:
              return insts_.Ground(Uniqueness::MostlyUnique);
            case K::Clobbered:
              return insts_.Ground(Uniqueness::Clobbered);
            case K::MostlyClobbered:
              return insts_.Ground(Uniqueness::MostlyClobbered);
            case K::Any:
              return insts_.Any();
            case K::UniqueAny:
              return insts_.Any(Uniqueness::Unique);
            case K::MostlyUniqueAny:
              return insts_.Any(Uniqueness::MostlyUnique);
            case K::ClobberedAny:
              return insts_.Any(Uniqueness::Clobbered);
            case K::MostlyClobberedAny:
              return insts_.Any(Uniqueness::MostlyClobbered);
            case K::NotReached:
              return insts_.NotReached();
            }
            return insts_.Ground();
          },
          [&](const parser::InstVar *var) {
            return Parameter(var->name, scope);
          },
          [&](const parser::BoundInst *bound) {
            std::vector<BoundAlternative> alternatives;
            std::vector<std::vector<const Inst *>> arguments;
            for (const parser::BoundFunctor *functor : bound->functors) {
              arguments.emplace_back();
              for (const parser::Inst &argument : functor->arguments) {
                arguments.back().push_back(Convert(argument, scope));
              }
            }
            for (std::size_t i{0}; i < bound->functors.size(); ++i) {
              const parser::BoundFunctor &functor{*bound->functors[i]};
              InstFunctor f{common::visit(
                  common::visitors{
                      [&](const parser::SymName *name) {
                        return InstFunctor{
                            result_.arena_.Save(llvm::join(name->qualifiers, ".")),
                            name->name,
                            static_cast<unsigned>(functor.arguments.size()),
                            false};
                      },
                      [&](const parser::Literal *literal) {
                        return InstFunctor{{}, literal->text, 0, true};
                      },
                  },
                  functor.name)};
              alternatives.push_back({f, arguments[i]});
            }
            return insts_.Bound(
                bound->unique ? Uniqueness::Unique : Uniqueness::Shared,
                alternatives);
          },
          [&](const parser::HigherOrderInst *higher) {
            std::vector<ArgMode> modes;
            for (const parser::Mode &mode : higher->arguments) {
              modes.push_back(Convert(mode, scope));
            }
            if (higher->result) {
              modes.push_back(Convert(*higher->result, scope));
            }
            return insts_.HigherOrder(
                higher->kind, higher->any, modes, higher->determinism);
          },
          [&](const parser::NamedInst *named) -> const Inst * {
            std::vector<const Inst *> arguments;
            for (const parser::Inst &argument : named->arguments) {
              arguments.push_back(Convert(argument, scope));
            }
            const Symbol *symbol{
                Named(*named->name, SymbolKind::Inst, arguments.size())};
            if (!symbol) {
              return insts_.Ground();
            }
            DefineInst(*symbol);
            if (bad_.count(symbol)) {
              if (own_) {
                messages_.Say(Source(*named->name),
                    "undefined inst %s/%u"_err_en_US, named->name->name.str(),
                    symbol->arity);
              }
              return insts_.Ground();
            }
            return insts_.Defined(*symbol, arguments);
          },
      },
      inst);
}

ArgMode ModeChecker::Convert(const parser::Mode &mode, InstScope &scope) {
  return common::visit(
      common::visitors{
          [&](const parser::MappingMode *mapping) {
            return ArgMode{Convert(mapping->initial, scope),
                Convert(mapping->final, scope)};
          },
          [&](const parser::NamedMode *named) {
            std::vector<const Inst *> arguments;
            for (const parser::Inst &argument : named->arguments) {
              arguments.push_back(Convert(argument, scope));
            }
            const Symbol *symbol{
                Named(*named->name, SymbolKind::Mode, arguments.size())};
            ArgMode error{insts_.Ground(), insts_.Ground()};
            if (!symbol) {
              return error;
            }
            std::optional<ArgMode> body{DefineMode(*symbol)};
            if (!body) {
              if (bad_.count(symbol) && own_) {
                messages_.Say(Source(*named->name),
                    "undefined mode %s/%u"_err_en_US, named->name->name.str(),
                    symbol->arity);
              }
              return error;
            }
            auto replace{[&](const Inst &parameter) {
              return parameter.number < arguments.size()
                  ? arguments[parameter.number]
                  : &parameter;
            }};
            return ArgMode{insts_.Substitute(body->initial, replace),
                insts_.Substitute(body->final, replace)};
          },
          [&](const parser::HigherOrderInst *higher) {
            // A higher-order inst used as a mode means in(Inst).
            const Inst *inst{Convert(parser::Inst{higher}, scope)};
            return ArgMode{inst, inst};
          },
      },
      mode);
}

std::optional<std::vector<ArgMode>> ModeChecker::LambdaModes(
    const Lambda &lambda) {
  InstScope scope;
  scope.id = ++scopes_;
  for (const std::optional<parser::Mode> &mode : lambda.modes) {
    if (mode) {
      Collect(*mode, scope);
    }
  }
  // Its inst variables are its own parameters, as a declaration's are.
  if (!scope.inconsistent.empty()) {
    return std::nullopt;
  }
  bool function{lambda.kind == parser::PredOrFunc::Function};
  std::vector<ArgMode> modes;
  for (std::size_t i{0}; i < lambda.arguments.size(); ++i) {
    if (i < lambda.modes.size() && lambda.modes[i]) {
      modes.push_back(Convert(*lambda.modes[i], scope));
    } else if (function && i + 1 == lambda.arguments.size()) {
      modes.push_back(ArgMode{insts_.Free(), insts_.Ground()});
    } else {
      modes.push_back(ArgMode{insts_.Ground(), insts_.Ground()});
    }
  }
  return modes;
}

void ModeChecker::DefineInst(const Symbol &symbol) {
  if (instState_.count(&symbol)) {
    return;
  }
  instState_[&symbol] = State::InProgress;
  const auto *defn{std::get_if<const parser::InstDefn *>(&symbol.declaration)};
  if (!defn) {
    instState_[&symbol] = State::Done;
    return;
  }
  bool saved{own_};
  own_ = Own(symbol);
  if ((*defn)->forType && !(*defn)->forType->name->candidates.empty()) {
    forTypes_[&symbol] = (*defn)->forType->name->candidates.front();
  }
  InstScope scope;
  scope.definition = true;
  scope.parameters = (*defn)->parameters;
  const Inst *body{Convert((*defn)->inst, scope)};
  if (!scope.freeParameter.empty()) {
    // mmc: "free inst parameter `J' on right hand side of inst
    // definition".
    if (own_) {
      messages_.Say(symbol.source,
          "inst parameter %s is not a parameter of the inst %s/%u"_err_en_US,
          scope.freeParameter.str(), symbol.name.str(), symbol.arity);
    }
    bad_.insert(&symbol);
  }
  insts_.SetBody(symbol, body);
  own_ = saved;
  instState_[&symbol] = State::Done;
}

std::optional<ArgMode> ModeChecker::DefineMode(const Symbol &symbol) {
  if (auto it{modeState_.find(&symbol)}; it != modeState_.end()) {
    if (it->second == State::Done) {
      auto found{modes_.find(&symbol)};
      return found == modes_.end() ? std::nullopt
                                   : std::optional<ArgMode>{found->second};
    }
    // It expands to itself: report it at the first of the cycle's
    // definitions, naming the next one.
    auto start{llvm::find(expanding_, &symbol)};
    std::vector<const Symbol *> cycle{start, expanding_.end()};
    if (llvm::none_of(cycle, [&](const Symbol *s) { return cyclic_.count(s); })) {
      cyclic_.insert(cycle.begin(), cycle.end());
      const Symbol &first{*cycle.front()};
      if (Own(first)) {
        // mmc: "mode name `m1'/0 expands to a mode containing itself
        // through `m2'/0".
        if (cycle.size() > 1) {
          messages_
              .Say(first.source,
                  "the mode %s/%u expands to a mode containing itself, through %s/%u"_err_en_US,
                  first.name.str(), first.arity, cycle[1]->name.str(),
                  cycle[1]->arity)
              .Attach(cycle[1]->source, "the definition of %s/%u is here"_because_en_US,
                  cycle[1]->name.str(), cycle[1]->arity);
        } else {
          messages_.Say(first.source,
              "the mode %s/%u expands to a mode containing itself"_err_en_US,
              first.name.str(), first.arity);
        }
      }
    }
    return std::nullopt;
  }
  const auto *defn{std::get_if<const parser::ModeDefn *>(&symbol.declaration)};
  if (!defn) {
    modeState_[&symbol] = State::Done;
    return std::nullopt;
  }
  modeState_[&symbol] = State::InProgress;
  expanding_.push_back(&symbol);
  bool saved{own_};
  own_ = Own(symbol);
  InstScope scope;
  scope.definition = true;
  scope.parameters = (*defn)->parameters;
  ArgMode body{Convert((*defn)->mode, scope)};
  if (!scope.freeParameter.empty()) {
    // mmc: "free inst parameter `I' on right hand side of mode
    // definition".
    if (own_) {
      messages_.Say(symbol.source,
          "inst parameter %s is not a parameter of the mode %s/%u"_err_en_US,
          scope.freeParameter.str(), symbol.name.str(), symbol.arity);
    }
    bad_.insert(&symbol);
  }
  own_ = saved;
  expanding_.pop_back();
  modeState_[&symbol] = State::Done;
  if (bad_.count(&symbol) || cyclic_.count(&symbol)) {
    return std::nullopt;
  }
  modes_[&symbol] = body;
  return body;
}

// An inst definition that is a named inst, which is a named inst, and so
// on, back to itself, never reaches a body.
void ModeChecker::CheckInstCycles() {
  llvm::DenseSet<const Symbol *> reported;
  for (const Symbol *symbol : result_.instDefinitions_) {
    std::vector<const Symbol *> chain{symbol};
    const Inst *body{insts_.Body(*symbol)};
    bool cycle{false};
    while (body && body->kind == InstKind::Defined && body->symbol) {
      if (body->symbol == symbol) {
        cycle = true;
        break;
      }
      if (llvm::is_contained(chain, body->symbol)) {
        break; // a cycle that does not pass through this one
      }
      chain.push_back(body->symbol);
      DefineInst(*body->symbol);
      body = insts_.Body(*body->symbol);
    }
    if (!cycle ||
        llvm::any_of(chain, [&](const Symbol *s) { return reported.count(s); })) {
      continue;
    }
    reported.insert(chain.begin(), chain.end());
    // mmc: "inst name `a'/0 expands to an inst containing itself through
    // `b'/0".
    if (chain.size() > 1) {
      messages_
          .Say(symbol->source,
              "the inst %s/%u expands to an inst containing itself, through %s/%u"_err_en_US,
              symbol->name.str(), symbol->arity, chain[1]->name.str(),
              chain[1]->arity)
          .Attach(chain[1]->source, "the definition of %s/%u is here"_because_en_US,
              chain[1]->name.str(), chain[1]->arity);
    } else {
      messages_.Say(symbol->source,
          "the inst %s/%u expands to an inst containing itself"_err_en_US,
          symbol->name.str(), symbol->arity);
    }
  }
}

//===----------------------------------------------------------------------===//
// Procedures
//===----------------------------------------------------------------------===//

ModeChecker::Declared ModeChecker::ConvertDeclaration(
    llvm::ArrayRef<parser::Mode> arguments, std::optional<parser::Mode> result,
    std::optional<parser::Inst> withInst,
    std::optional<parser::Determinism> determinism,
    llvm::ArrayRef<const parser::InstConstraint *> constraints,
    parser::CharBlock where) {
  InstScope scope;
  scope.id = ++scopes_;
  for (const parser::InstConstraint *constraint : constraints) {
    InstScope inner;
    const Inst *inst{Convert(constraint->inst, inner)};
    // mmc keeps the first of a variable's constraints in the list, and
    // reports none that differ (see the design document's register).
    if (!llvm::is_contained(scope.names, constraint->var)) {
      scope.names.push_back(constraint->var);
      scope.constraints.push_back(inst);
    }
  }
  for (const parser::Mode &mode : arguments) {
    Collect(mode, scope);
  }
  if (result) {
    Collect(*result, scope);
  }
  if (withInst) {
    Collect(*withInst, scope);
  }
  Declared declared;
  declared.determinism = determinism;
  for (const parser::Mode &mode : arguments) {
    declared.modes.push_back(Convert(mode, scope));
  }
  if (withInst) {
    // `with_inst` Inst adds the arguments of a higher-order inst
    // (reference manual, "Predicate and function mode declarations").
    const Inst *inst{Convert(*withInst, scope)};
    for (unsigned steps{0}; inst->kind == InstKind::Defined && steps < 100;
         ++steps) {
      inst = insts_.Expand(inst);
    }
    if (inst->kind == InstKind::HigherOrder) {
      declared.modes.insert(
          declared.modes.end(), inst->modes.begin(), inst->modes.end());
      if (!declared.determinism) {
        declared.determinism = inst->determinism;
      }
    }
  }
  if (result) {
    declared.modes.push_back(Convert(*result, scope));
  }
  declared.parameters = scope.made;
  if (!scope.inconsistent.empty()) {
    // mmc: "inconsistent constraints on inst variable I in predicate
    // declaration".
    messages_.Say(where,
        "the inst parameter %s has inconsistent constraints in this declaration"_err_en_US,
        scope.inconsistent.str());
  }
  return declared;
}

const Symbol *ModeChecker::ModeDeclTarget(const parser::ModeDecl &decl,
    const Symbol *typeclass, const ModuleSymbols &own) {
  parser::PredOrFunc kind{decl.result ? parser::PredOrFunc::Function
                                      : parser::PredOrFunc::Predicate};
  SymbolKind symbolKind{decl.result ? SymbolKind::Function
                                    : SymbolKind::Predicate};
  unsigned arity{static_cast<unsigned>(decl.arguments.size())};
  std::string qualifier{llvm::join(decl.name->qualifiers, ".")};
  llvm::SmallVector<unsigned, 2> others;
  // A function of that arity, for a predicate mode, or the other way.
  bool otherKind{false};
  if (qualifier.empty() || qualifier == own.name()) {
    for (const Symbol *symbol : own.Lookup(decl.name->name)) {
      if (symbol->owner != typeclass) {
        continue;
      }
      if (symbol->kind == SymbolKind::Predicate ||
          symbol->kind == SymbolKind::Function) {
        otherKind |= symbol->kind != symbolKind && symbol->arity == arity;
      }
      if (symbol->kind != symbolKind) {
        continue;
      }
      // `with_inst` adds arguments the mode declaration does not list.
      if (symbol->arity == arity ||
          (decl.withInst && symbol->arity > arity)) {
        return symbol;
      }
      others.push_back(symbol->arity);
    }
  }
  // mmc: "`:- mode' declaration for predicate `q'/1 without corresponding
  // `:- pred' declaration. However, a predicate of that name does exist
  // with arity 1."
  std::string hint;
  if (!others.empty()) {
    hint = std::string{"; there is a "} + Keyword(kind) + " of that name with arity " +
        llvm::join(llvm::map_range(others,
                       [](unsigned a) { return std::to_string(a); }),
            " and ");
  }
  if (otherKind) {
    hint += decl.result
        ? "; there is a pred of that name and arity, whose mode declaration has no result"
        : "; there is a func of that name and arity, whose mode declaration has a result, as in `:- mode f(in) = out'";
  }
  if (!own_) {
    return nullptr;
  }
  messages_.Say(Source(*decl.name),
      "mode declaration for %s %s/%u without a %s declaration%s"_err_en_US,
      Keyword(kind), decl.name->name.str(), arity, Keyword(kind), hint);
  return nullptr;
}

// An inst that is declared for one type constructor must be applied to
// values of that type (reference manual, "Insts, modes, and mode
// definitions"). mmc checks each argument's inst, and the insts within it
// as far as the argument's type gives their types, reporting the first
// mismatch of each argument.
void ModeChecker::CheckForTypes(const Symbol &symbol,
    llvm::ArrayRef<ArgMode> modes, parser::CharBlock where) {
  const Signature *signature{types_.SignatureOf(symbol)};
  if (!signature) {
    return;
  }
  llvm::ArrayRef<const Type *> arguments{signature->arguments};
  for (std::size_t i{0}; i < modes.size() && i < arguments.size(); ++i) {
    llvm::DenseSet<std::pair<const Inst *, const Type *>> seen;
    // The named inst whose expansion is being walked, if any.
    const Symbol *expanding{nullptr};
    std::function<bool(const Inst *, const Type *)> walk{
        [&](const Inst *inst, const Type *type) {
          if (!type || type->kind != TypeKind::Named ||
              type->constructor->name == "<error>" ||
              !seen.insert({inst, type}).second) {
            return false;
          }
          if (inst->kind == InstKind::Defined && inst->symbol) {
            const Symbol *forType{forTypes_.lookup(inst->symbol)};
            if (forType && forType != type->constructor) {
              // mmc: "the user defined inst `iy'/0 is declared to be
              // applicable only to values of type `list.list'/1, but here
              // it is applied to values of type `int'/0", after "in the
              // expansion of the user-defined inst `wrapped'/0" if within
              // one.
              std::string within;
              if (expanding) {
                within = ", within the inst " + expanding->module.str() + '.' +
                    expanding->name.str() + '/' +
                    std::to_string(expanding->arity) + ',';
              }
              messages_.Say(where,
                  "the inst %s.%s/%u is for values of type %s.%s/%u, but in argument %u of %s%s it is applied to values of type %s"_err_en_US,
                  inst->symbol->module.str(), inst->symbol->name.str(),
                  inst->symbol->arity, forType->module.str(),
                  forType->name.str(), forType->arity,
                  static_cast<unsigned>(i + 1), Text(symbol), within,
                  TypeToString(*type));
              return true;
            }
            const Symbol *saved{expanding};
            if (!expanding) {
              expanding = inst->symbol;
            }
            bool found{walk(insts_.Expand(inst), type)};
            expanding = saved;
            return found;
          }
          if (inst->kind != InstKind::Bound) {
            return false;
          }
          std::vector<ConstructorTypes> constructors;
          if (!constructors_.Constructors(type, constructors)) {
            return false;
          }
          for (const BoundAlternative &alternative : inst->functors) {
            for (const ConstructorTypes &constructor : constructors) {
              if (constructor.functor.name != alternative.functor.name ||
                  constructor.functor.arity != alternative.functor.arity) {
                continue;
              }
              for (std::size_t j{0}; j < alternative.arguments.size(); ++j) {
                if (walk(alternative.arguments[j], constructor.arguments[j])) {
                  return true;
                }
              }
            }
          }
          return false;
        }};
    for (const Inst *inst : {modes[i].initial, modes[i].final}) {
      if (walk(inst, arguments[i])) {
        break;
      }
    }
  }
}

void ModeChecker::Run(const parser::ParseTree &tree) {
  result_.constructors_ = std::make_unique<ConstructorTable>(resolution_, types_);
  result_.ops_ = std::make_unique<InstOps>(insts_, *result_.constructors_);
  const ModuleSymbols &own{resolution_.module()};
  for (const Symbol *symbol : own.symbols()) {
    if (symbol->kind == SymbolKind::Inst) {
      result_.instDefinitions_.push_back(symbol);
      if (ReservedInst(symbol->name)) {
        // mmc: "the inst name `ground' is reserved for the Mercury
        // implementation".
        messages_.Say(symbol->source,
            "the inst name %s is reserved for the Mercury implementation"_err_en_US,
            symbol->name.str());
      }
      DefineInst(*symbol);
    } else if (symbol->kind == SymbolKind::Mode) {
      if (ReservedMode(symbol->name)) {
        messages_.Say(symbol->source,
            "the mode name %s is reserved for the Mercury implementation"_err_en_US,
            symbol->name.str());
      }
      DefineMode(*symbol);
    }
  }
  CheckInstCycles();

  // The procedures of the imported interfaces' predicates and functions,
  // for calls, and then the module's own.
  for (const ModuleInterface *interface : resolution_.interfaces()) {
    if (const ModuleSymbols *symbols{resolution_.Module(interface->name)}) {
      own_ = false;
      tokensNow_ = interface->tokens;
      BuildProcedures(interface->items, *symbols);
    }
  }
  own_ = true;
  tokensNow_ = &tokens_;
  BuildProcedures(tree.items, own);
  for (const Procedure &procedure : result_.procedures_) {
    result_.bySymbol_[procedure.symbol].push_back(&procedure);
  }
}

void ModeChecker::BuildProcedures(
    llvm::ArrayRef<parser::Item> items, const ModuleSymbols &symbols) {
  // The mode declarations of each predicate and function, with the
  // section each is in.
  struct ModeItem {
    const parser::ModeDecl *decl;
    bool interface;
  };
  llvm::DenseMap<const Symbol *, std::vector<ModeItem>> modeDecls;
  bool interface{true};
  auto addMode{[&](const parser::ModeDecl &decl, const Symbol *typeclass) {
    if (const Symbol *target{ModeDeclTarget(decl, typeclass, symbols)}) {
      modeDecls[target].push_back({&decl, interface});
    }
  }};
  for (const parser::Item &item : items) {
    if (const auto *section{std::get_if<const parser::SectionDecl *>(&item)}) {
      interface = (*section)->section == parser::Section::Interface;
    } else if (const auto *decl{std::get_if<const parser::ModeDecl *>(&item)}) {
      addMode(**decl, nullptr);
    } else if (const auto *typeclass{
                   std::get_if<const parser::TypeclassDecl *>(&item)}) {
      const Symbol *symbol{nullptr};
      for (const Symbol *s : symbols.Lookup((*typeclass)->name->name)) {
        if (s->kind == SymbolKind::Typeclass &&
            s->arity == (*typeclass)->parameters.size()) {
          symbol = s;
        }
      }
      for (const parser::ClassMethod &method : (*typeclass)->methods) {
        if (const auto *decl{std::get_if<const parser::ModeDecl *>(&method)}) {
          if (symbol) {
            addMode(**decl, symbol);
          }
        }
      }
    }
  }

  for (const Symbol *symbol : symbols.symbols()) {
    if (symbol->kind != SymbolKind::Predicate &&
        symbol->kind != SymbolKind::Function) {
      continue;
    }
    const auto *decl{std::get_if<const parser::PredDecl *>(&symbol->declaration)};
    if (!decl) {
      if (!own_) {
      } else if (symbol->kind == SymbolKind::Function) {
        // A function without a declaration has the default mode, as in
        // mmc, where only a predicate's modes are inferred.
        std::vector<ArgMode> modes(symbol->arity,
            ArgMode{insts_.Ground(), insts_.Ground()});
        modes.push_back(ArgMode{insts_.Free(), insts_.Ground()});
        result_.procedures_.push_back(Procedure{symbol, 0,
            result_.arena_.SaveArray(llvm::ArrayRef<ArgMode>{modes}),
            parser::Determinism::Det, {}});
      } else {
        // Its modes are inferred from its calls (Overton, 3.4.2).
        result_.inferable_.insert(symbol);
      }
      continue;
    }
    std::vector<Procedure> procedures;
    std::vector<parser::CharBlock> sources;
    // As in mmc, modes are the same if they are with each inst variable
    // replaced by its constraint, so in(I) is in.
    auto erased{[&](llvm::ArrayRef<ArgMode> modes) {
      auto constraint{[&](const Inst &parameter) {
        return parameter.constraint ? parameter.constraint : &parameter;
      }};
      std::vector<ArgMode> result;
      for (const ArgMode &mode : modes) {
        result.push_back({insts_.Substitute(mode.initial, constraint),
            insts_.Substitute(mode.final, constraint)});
      }
      return result;
    }};
    auto add{[&](Declared declared, parser::CharBlock where) {
      for (std::size_t i{0}; i < procedures.size(); ++i) {
        if (erased(procedures[i].modes) == erased(declared.modes)) {
          // mmc: "In mode declarations for predicate `p'/1: error:
          // duplicate mode declaration. Here is the conflicting mode
          // declaration."
          if (own_) {
            messages_
                .Say(sources[i], "%s has the same mode declared twice"_err_en_US,
                    Text(*symbol))
                .Attach(where, "the other declaration of the mode is here"_because_en_US);
          }
          return;
        }
      }
      if (own_) {
        CheckForTypes(*symbol, declared.modes, where);
      }
      procedures.push_back(Procedure{symbol,
          static_cast<unsigned>(procedures.size()),
          result_.arena_.SaveArray(llvm::ArrayRef<ArgMode>{declared.modes}),
          declared.determinism, where,
          result_.arena_.SaveArray(
              llvm::ArrayRef<const Inst *>{declared.parameters})});
      sources.push_back(where);
    }};
    const parser::PredDecl &pred{**decl};
    bool combined{(pred.result ? pred.result->mode.has_value() : false) ||
        llvm::any_of(pred.arguments,
            [](const parser::TypeAndMode *a) { return a->mode.has_value(); }) ||
        pred.withInst ||
        (pred.arguments.empty() && !pred.result && pred.determinism)};
    if (combined) {
      std::vector<parser::Mode> arguments;
      for (const parser::TypeAndMode *argument : pred.arguments) {
        if (argument->mode) {
          arguments.push_back(*argument->mode);
        }
      }
      std::optional<parser::Mode> result;
      if (pred.result && pred.result->mode) {
        result = *pred.result->mode;
      }
      llvm::SmallVector<const parser::InstConstraint *, 2> constraints;
      for (const parser::Constraint &constraint : pred.universalConstraints) {
        if (const auto *inst{
                std::get_if<const parser::InstConstraint *>(&constraint)}) {
          constraints.push_back(*inst);
        }
      }
      Declared declared{ConvertDeclaration(arguments, result, pred.withInst,
          pred.determinism, constraints, symbol->source)};
      // A `with_inst` inst gives its determinism.
      if (own_ && symbol->exported && !declared.determinism) {
        // mmc: "no determinism declaration for exported predicate `q'/1".
        messages_.Say(symbol->source,
            "the exported %s has no determinism declaration"_err_en_US,
            Text(*symbol));
      }
      add(std::move(declared), symbol->source);
    }
    for (const ModeItem &item : modeDecls.lookup(symbol)) {
      const parser::ModeDecl &mode{*item.decl};
      parser::CharBlock where{Source(*mode.name)};
      if (own_ && combined) {
        // mmc: "predicate `p'/1 has its predicate declaration combined
        // with a mode declaration, so it may not have a separate mode
        // declaration".
        messages_.Say(where,
            "%s has its modes in its %s declaration, so it may not have a separate mode declaration"_err_en_US,
            Text(*symbol), Keyword(pred.kind));
      }
      if (own_ && item.interface != symbol->exported) {
        // mmc: "mode declaration in the implementation section for
        // predicate `r'/1, whose predicate declaration is in the
        // interface".
        messages_.Say(where,
            "this mode declaration for %s is in the %s section, but its %s declaration is in the %s"_err_en_US,
            Text(*symbol),
            item.interface ? "interface" : "implementation", Keyword(pred.kind),
            symbol->exported ? "interface" : "implementation");
      }
      std::optional<parser::Mode> result;
      if (mode.result) {
        result = *mode.result;
      }
      Declared declared{ConvertDeclaration(mode.arguments, result,
          mode.withInst, mode.determinism, mode.constraints, where)};
      if (own_ && item.interface == symbol->exported && symbol->exported &&
          !declared.determinism) {
        // mmc: "no determinism declaration for exported predicate `p'/1".
        messages_.Say(where,
            "the exported %s has no determinism declaration in this mode declaration"_err_en_US,
            Text(*symbol));
      }
      add(std::move(declared), where);
    }
    if (procedures.empty() && !combined && !modeDecls.count(symbol)) {
      if (symbol->kind == SymbolKind::Function) {
        // The default mode: every argument in, the result out, det unless
        // the declaration gives a determinism (reference manual,
        // "Determinism categories").
        std::vector<ArgMode> modes(symbol->arity,
            ArgMode{insts_.Ground(), insts_.Ground()});
        modes.push_back(ArgMode{insts_.Free(), insts_.Ground()});
        parser::Determinism determinism{
            pred.determinism.value_or(parser::Determinism::Det)};
        procedures.push_back(Procedure{symbol, 0,
            result_.arena_.SaveArray(llvm::ArrayRef<ArgMode>{modes}),
            determinism, {}});
      } else if (!own_) {
      } else if (symbol->exported) {
        // mmc: "no mode declaration for exported predicate `m.p'/1", even
        // with --infer-modes.
        messages_.Say(symbol->source,
            "the exported %s has no mode declaration"_err_en_US, Text(*symbol));
      } else if (!options_.inferModes) {
        // mmc: "no mode declaration for predicate `p'/1".
        messages_.Say(symbol->source, "%s has no mode declaration"_err_en_US,
            Text(*symbol));
      } else {
        // With -finfer-modes, as mmc's --infer-modes, its modes are
        // inferred from its calls, as an undeclared predicate's are.
        result_.inferable_.insert(symbol);
      }
    }
    if (own_) {
      result_.procedures_.insert(
          result_.procedures_.end(), procedures.begin(), procedures.end());
    } else {
      std::vector<const Procedure *> &list{result_.bySymbol_[symbol]};
      for (const Procedure &procedure : procedures) {
        list.push_back(&result_.imported_.emplace_back(procedure));
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Procedures' clauses
//===----------------------------------------------------------------------===//

std::string ProcedureToString(const Procedure &procedure) {
  std::string text{procedure.symbol->name.str()};
  bool function{procedure.symbol->kind == SymbolKind::Function};
  llvm::ArrayRef<ArgMode> arguments{procedure.modes};
  if (function && !arguments.empty()) {
    arguments = arguments.drop_back();
  }
  if (!arguments.empty()) {
    text += '(';
    llvm::interleave(
        arguments, [&](const ArgMode &mode) { text += ArgModeToString(mode); },
        [&] { text += ", "; });
    text += ')';
  }
  if (function && !procedure.modes.empty()) {
    text += " = " + ArgModeToString(procedure.modes.back());
  }
  return text;
}

namespace {

// A variable's name in messages: a program variable's own, a head
// variable's HeadVar__N, as in mmc, and the others' V_N.
std::string VarName(const ClauseForm &clause, VarId var) {
  const VarInfo &info{clause.vars[var]};
  if (info.kind == VarKind::Program) {
    return info.name.str();
  }
  if (info.kind == VarKind::State) {
    return info.name.str() + '!' + std::to_string(info.version);
  }
  if (const auto *it{llvm::find(clause.head, var)}; it != clause.head.end()) {
    return "HeadVar__" + std::to_string(it - clause.head.begin() + 1);
  }
  return "V_" + std::to_string(var);
}

std::string FunctorText(const Functor &functor) {
  switch (functor.kind) {
  case FunctorKind::Name:
    return functor.name->name.str();
  case FunctorKind::Cons:
    return "'[|]'";
  case FunctorKind::Nil:
    return "[]";
  case FunctorKind::Tuple:
    return "{}";
  case FunctorKind::Literal:
    return functor.literal->text.str();
  case FunctorKind::ImplementationLiteral:
    return "$" + functor.implementation->name.str();
  case FunctorKind::Coerce:
    return "coerce";
  case FunctorKind::Lambda:
    return "lambda expression";
  }
  return "";
}

std::string TermText(const ClauseForm &clause, const Unify &unify) {
  if (unify.isVar) {
    return VarName(clause, unify.var);
  }
  std::string text{FunctorText(unify.functor)};
  if (!unify.arguments.empty()) {
    text += '(';
    llvm::interleave(
        unify.arguments, [&](VarId var) { text += VarName(clause, var); },
        [&] { text += ", "; });
    text += ')';
  }
  return text;
}

parser::TokenRange TokensOf(const Goal &goal) {
  return std::visit([](const auto *g) { return g->tokens; }, goal);
}

unsigned NumberOf(const Goal &goal) {
  return std::visit([](const auto *g) { return g->number; }, goal);
}

} // namespace

class ProcedureChecker {
public:
  ProcedureChecker(ModeChecking &result, ModeChecker &declarations,
      const NameResolution &resolution, TypeChecking &types, InstOps &ops,
      const parser::TokenSequence &tokens, parser::Messages &messages,
      const ModeCheckOptions &options)
      : result_{result}, declarations_{declarations}, insts_{result.insts_},
        resolution_{resolution}, types_{types}, ops_{ops}, tokens_{tokens},
        out_{messages}, options_{options} {}

  void Run();

private:
  using InstMap = std::vector<const Inst *>;
  // The variables used after a goal: those of the goals still to be run in
  // the conjunctions around it, and the head's.
  using Live = std::vector<bool>;
  enum class Outcome { Ok, Error, Unsupported };
  struct Result {
    Outcome outcome{Outcome::Ok};
    parser::CharBlock where{};
    std::string text{};
    Unchecked why{Unchecked::None};
    // For an error, the variables whose insts would have to change for the
    // goal to run; all of the goal's if none.
    std::vector<VarId> waits{};
    std::vector<std::pair<parser::CharBlock, std::string>> notes{};
    // For an error, that no variable can make the goal run, as for a
    // lambda expression with a mode error in it; then `waits` is empty.
    bool never{false};
    // Whether the error is in a lambda expression.
    bool lambda{false};
  };
  static Result Ok() { return {}; }
  static Result Unsupported(Unchecked why) {
    return {Outcome::Unsupported, {}, {}, why};
  }
  Result Error(
      const Goal &goal, std::string text, std::vector<VarId> waits = {}) {
    return {Outcome::Error, Source(TokensOf(goal)),
        InClause() + ": " + text,
        Unchecked::None, std::move(waits)};
  }

  parser::CharBlock Source(parser::TokenRange range) const {
    return {tokens_.tokens[range.first].spelling.begin(),
        tokens_.tokens[range.last].spelling.end()};
  }
  // The position of a goal: its first token.
  parser::CharBlock Start(parser::TokenRange range) const {
    return tokens_.tokens[range.first].spelling;
  }
  const Type *TypeOf(VarId var) const { return clause_->vars[var]; }
  std::string Name(VarId var) const { return VarName(*form_, var); }
  std::string InstText(const Inst *inst) const { return InstToString(*inst); }

  Result CheckClause(const ClauseTypes &, const Procedure &, ClauseModes &);
  void RunPass();
  std::vector<ArgMode> InferredModes(
      llvm::ArrayRef<VarId> arguments, const InstMap &, const Live &);
  const Procedure *Infer(const Symbol &, llvm::ArrayRef<ArgMode>);
  std::string RejectedText(const Procedure &);
  Result RejectedCall(const Goal &, const Symbol &callee,
      const Procedure &rejected, llvm::ArrayRef<VarId> arguments,
      const InstMap &);
  std::pair<Result, std::vector<ClauseModes>> CheckOwn(const Procedure &);
  void Record(const Procedure &, std::pair<Result, std::vector<ClauseModes>>);
  Result CheckProcedure(const Procedure &,
      llvm::ArrayRef<const ClauseTypes *>, std::vector<ClauseModes> &checked);
  std::string InClause() const;
  Result CheckClauses(llvm::ArrayRef<ClauseModes>);
  const ClauseTypes &WithHeadUnifications(const ClauseTypes &);
  Result Check(const Goal &, InstMap &, const Live &);
  Result CheckLocked(const Goal &, InstMap &, const Live &);
  Result CheckGoal(const Goal &, InstMap &, const Live &);
  Result Disjunction(const Goal &, const Disj &, InstMap &, const Live &);
  Result IfThenElseGoal(
      const Goal &, const IfThenElse &, InstMap &, const Live &);
  Result Negation(const Goal &, const Not &, InstMap &, const Live &);
  // The error for a negated context that a nonlocal with an any part
  // appears in, outside a purity promise.
  std::optional<Result> AnyInNegatedContext(const Goal &,
      llvm::ArrayRef<VarId> vars, const InstMap &, const char *context,
      const char *part);
  Result Merge(const Goal &, const char *what, llvm::ArrayRef<InstMap>,
      llvm::ArrayRef<parser::CharBlock> branches, InstMap &, const Live &);
  Result Mismatch(const char *what, const std::string &name,
      llvm::ArrayRef<const Inst *>, llvm::ArrayRef<parser::CharBlock> branches,
      const Type *, bool first = true);
  bool HeadLive(const Procedure &, std::size_t i) const;
  // Where a clause's head ends, at its `:-`, where mmc puts the clause as a
  // branch of the predicate's disjunction.
  parser::CharBlock HeadEnd(const ClauseForm &) const;
  Result Conjunction(const Goal &conj, llvm::ArrayRef<Goal>, InstMap &,
      const Live &);
  Result CheckUnify(const Unify &, const Goal &, InstMap &, const Live &);
  Result CheckVarUnify(const Unify &, const Goal &, InstMap &, const Live &);
  Result CheckCall(const Goal &, const Symbol &, llvm::ArrayRef<VarId>,
      bool function, InstMap &, GoalModes &, const Live &);
  // An error if the modes clobber an argument that is used after the
  // call.
  std::optional<Result> Clobbers(const Goal &, llvm::ArrayRef<ArgMode>,
      llvm::ArrayRef<VarId> arguments, const Live &,
      llvm::function_ref<std::string(std::size_t)> where);
  Result CheckLambda(const Unify &, const Goal &, InstMap &, const Live &);
  Result CheckCoerce(const Unify &, const Goal &, InstMap &, const Live &);
  // s =< t, by the visible subtype definitions.
  bool IsSubtype(const Type *s, const Type *t, unsigned depth = 0);
  Result CheckClosure(const Unify &, const Goal &, const Symbol &, InstMap &);
  Result CheckHigherOrderCall(const Goal &, const HOCall &, InstMap &,
      const Live &);
  Result HigherOrderUnification(
      const Goal &, VarId, const std::string &other);
  // That the first of `waits`, which are not ground, is the nonlocal of a
  // lambda expression or an argument of a higher-order term.
  Result NonlocalError(const Goal &, const InstMap &, std::vector<VarId> waits);
  // The first argument that does not fit its mode's initial inst, if any.
  // With `implied`, a bound argument fits an output too: the call outputs
  // to a fresh variable, which is then unified with it (an implied mode,
  // Overton 3.3.2). The argument may have free parts, of variables not
  // used again, which the unification binds, as in mmc.
  // A callee's modes with the inst parameters bound by the arguments.
  std::vector<ArgMode> Instantiate(llvm::ArrayRef<ArgMode>,
      llvm::ArrayRef<VarId> arguments, const InstMap &);
  std::optional<std::size_t> Misfit(llvm::ArrayRef<ArgMode>,
      llvm::ArrayRef<VarId>, const InstMap &, bool implied);
  // The error for the first argument that does not fit its mode, which
  // waits for it and for the first that does not fit an implied mode, as
  // in mmc; `where` names argument i.
  Result ArgumentError(const Goal &, llvm::ArrayRef<ArgMode>,
      llvm::ArrayRef<VarId>, const InstMap &,
      llvm::function_ref<std::string(std::size_t)> where);
  // The arguments' insts after a call in these modes.
  void Pass(llvm::ArrayRef<ArgMode>, llvm::ArrayRef<VarId>, InstMap &);
  // An expected inst as mmc prints it: ground for a higher-order function
  // as the default function inst.
  std::string Expected(const Inst *expected, const Inst *actual);
  // That argument i, `name`, has the wrong final inst.
  // `subject` is "argument 1, X," or "argument 1 of the higher-order
  // term".
  std::string FinalError(const std::string &subject, const Inst *final,
      const Inst *expected, const Type *);
  // Whether an argument of this mode is used after the call or clause.
  bool LiveAfter(const ArgMode &) const;
  llvm::ArrayRef<const Procedure *> ProceduresFor(const Symbol &);
  Result CheckFinal(llvm::ArrayRef<const Inst *> final, bool clauses);
  static void Flatten(const Goal &, std::vector<Goal> &);

  ModeChecking &result_;
  ModeChecker &declarations_;
  InstContext &insts_;
  const NameResolution &resolution_;
  TypeChecking &types_;
  InstOps &ops_;
  const parser::TokenSequence &tokens_;
  parser::Messages &out_;
  const ModeCheckOptions &options_;
  // The messages of the module's procedures, kept for the last pass of
  // mode inference.
  parser::Messages pending_;
  parser::Messages &messages_{pending_};

  // The clauses of each predicate and function, and of each instance
  // method.
  llvm::DenseMap<const Symbol *, std::vector<const ClauseTypes *>> byPredicate_;
  std::vector<const MethodDefinition *> methodOrder_;
  llvm::DenseMap<const MethodDefinition *, std::vector<const ClauseTypes *>>
      byMethod_;
  // Mode inference: whether a pass made a procedure or changed an
  // inferred one, how each inferred procedure's check ended in the pass
  // before, and the final insts of the one being checked.
  bool changed_{false};
  llvm::DenseMap<const Procedure *, Outcome> outcomes_;
  std::vector<const Inst *> inferredFinal_;
  // The error of each rejected inferred procedure, reported with that of a
  // call that no other mode can run.
  llvm::DenseMap<const Procedure *, std::pair<parser::CharBlock, std::string>>
      rejections_;

  // The clause being checked, and for which procedure.
  const ClauseTypes *clause_{nullptr};
  const ClauseForm *form_{nullptr};
  const Procedure *procedure_{nullptr};
  std::vector<GoalModes> goals_;
  // By goal number, the insts of its nonlocals before it last ran.
  std::vector<std::vector<const Inst *>> befores_;
  // The goal run last, where an error in the final insts is reported, as
  // in mmc.
  parser::CharBlock last_{};
  // By variable, how often a construction put it in a term while free.
  std::vector<unsigned> touched_;
  // How often a goal checked left its insts not reached.
  unsigned unreachables_{0};
  // By variable, whether it is a nonlocal of a negation or of an
  // if-then-else whose condition is being checked, which may not bind it.
  enum class Lock { None, Negation, Condition };
  std::vector<Lock> locks_;
  // How many promise_pure, promise_semipure or promise_impure scopes the
  // goal being checked is in.
  unsigned promised_{0};
  // The head unifications of WithHeadUnifications.
  llvm::DenseSet<const Unify *> heads_;
  // Warnings of the goals run, reported once the clause is checked.
  std::vector<std::pair<parser::CharBlock, std::string>> warnings_;
  // The instance method whose clauses are being checked, if any.
  const MethodDefinition *method_{nullptr};
  // The procedures of fields, made when first called.
  llvm::DenseMap<const Symbol *, const Procedure *> fields_;
};

void ProcedureChecker::Run() {
  for (const ClauseTypes &clause : types_.clauses()) {
    if (clause.method) {
      auto &own{byMethod_[clause.method]};
      if (own.empty()) {
        methodOrder_.push_back(clause.method);
      }
      own.push_back(&clause);
    } else if (clause.definition) {
      byPredicate_[clause.definition].push_back(&clause);
    }
  }
  // Mode inference (Overton, 3.4.2): a call of a predicate without a
  // `:- pred` declaration makes a procedure for it, whose final insts are
  // not_reached until its clauses are checked. The module is checked again
  // until no pass makes a procedure or changes one's final insts, or, as
  // in mmc, one finds a mode error in a procedure not inferred, and the
  // messages are those of the last pass.
  for (unsigned pass{1};; ++pass) {
    pending_.clear();
    result_.unchecked_.clear();
    result_.errors_.clear();
    result_.clauses_.clear();
    result_.methods_.clear();
    changed_ = false;
    RunPass();
    if (!changed_ ||
        llvm::any_of(result_.errors_,
            [](const Procedure *p) { return !p->inferred; }) ||
        llvm::any_of(result_.methods_,
            [](const MethodModes &method) { return method.error; })) {
      break;
    }
    if (pass >= options_.inferenceIterationLimit) {
      // mmc: "mode analysis iteration limit exceeded".
      const Procedure &first{result_.inferred_.front()};
      out_.Say(first.source,
          "mode inference did not reach a fixpoint in %u passes"_err_en_US,
          pass);
      break;
    }
  }
  out_.Annex(std::move(pending_));
}

void ProcedureChecker::RunPass() {
  for (const Procedure &procedure : result_.procedures_) {
    Record(procedure, CheckOwn(procedure));
  }
  // An instance's method clauses are checked in each of the method's
  // modes, as mmc does, with the instance's types.
  for (const MethodDefinition *method : methodOrder_) {
    method_ = method;
    for (const Procedure *procedure : result_.ProceduresOf(*method->method)) {
      MethodModes &modes{result_.methods_.emplace_back()};
      modes.method = method;
      modes.procedure = procedure;
      Result r{CheckProcedure(*procedure, byMethod_[method], modes.clauses)};
      if (r.outcome == Outcome::Unsupported) {
        modes.unchecked = r.why;
      }
      modes.error = r.outcome == Outcome::Error;
      if (r.outcome != Outcome::Ok) {
        modes.clauses.clear();
      }
    }
  }
  method_ = nullptr;
  // The inferred procedures, those that this pass makes included.
  for (std::size_t i{0}; i < result_.inferred_.size(); ++i) {
    Procedure &procedure{result_.inferred_[i]};
    inferredFinal_.clear();
    std::pair<Result, std::vector<ClauseModes>> checked{CheckOwn(procedure)};
    Outcome outcome{checked.first.outcome};
    if (outcome == Outcome::Ok && !inferredFinal_.empty()) {
      // Each argument that the call did not clobber gets its final inst
      // from the clauses, normalised.
      std::vector<ArgMode> modes{procedure.modes.begin(), procedure.modes.end()};
      for (std::size_t j{0}; j < modes.size(); ++j) {
        if (LiveAfter(modes[j])) {
          modes[j].final = inferredFinal_[j];
        }
      }
      if (!llvm::equal(modes, procedure.modes, [](const ArgMode &a, const ArgMode &b) {
            return a.initial == b.initial && a.final == b.final;
          })) {
        procedure.modes = result_.arena_.SaveArray(llvm::ArrayRef<ArgMode>{modes});
        changed_ = true;
      }
    }
    if (outcome != Outcome::Error) {
      rejections_.erase(&procedure);
    }
    auto [it, inserted]{outcomes_.try_emplace(&procedure, outcome)};
    if (!inserted && it->second != outcome) {
      it->second = outcome;
      changed_ = true;
    }
    Record(procedure, std::move(checked));
  }
}

// A procedure's own clauses, checked.
std::pair<ProcedureChecker::Result, std::vector<ClauseModes>>
ProcedureChecker::CheckOwn(const Procedure &procedure) {
  auto found{byPredicate_.find(procedure.symbol)};
  llvm::ArrayRef<const ClauseTypes *> own;
  if (found != byPredicate_.end()) {
    own = found->second;
  }
  std::vector<ClauseModes> checked;
  Result r{CheckProcedure(procedure, own, checked)};
  return {std::move(r), std::move(checked)};
}

void ProcedureChecker::Record(const Procedure &procedure,
    std::pair<Result, std::vector<ClauseModes>> checked) {
  const Result &r{checked.first};
  if (r.outcome == Outcome::Unsupported) {
    result_.unchecked_[&procedure] = r.why;
  } else if (r.outcome == Outcome::Error) {
    result_.errors_.insert(&procedure);
  } else {
    result_.clauses_[&procedure] = std::move(checked.second);
  }
}

// A procedure's clauses, checked: its first mode error is reported, as in
// mmc, and those it checked are in `checked`.
ProcedureChecker::Result ProcedureChecker::CheckProcedure(
    const Procedure &procedure, llvm::ArrayRef<const ClauseTypes *> own,
    std::vector<ClauseModes> &checked) {
  // mmc checks modes only after the types are right.
  if (llvm::any_of(own, [](const ClauseTypes *c) { return c->typeError; })) {
    return Unsupported(Unchecked::TypeErrors);
  }
  if (llvm::any_of(
          own, [](const ClauseTypes *c) { return c->clause->modeSpecific; })) {
    return Unsupported(Unchecked::ModeSpecificClauses);
  }
  Result r{Ok()};
  for (std::size_t n{0}; n < own.size(); ++n) {
    const ClauseTypes *original{own[n]};
    const ClauseTypes *clause{
        own.size() > 1 ? &WithHeadUnifications(*original) : original};
    ClauseModes modes{clause, &procedure, {}, {}};
    warnings_.clear();
    r = CheckClause(*clause, procedure, modes);
    if (r.outcome != Outcome::Unsupported) {
      for (const auto &[where, text] : warnings_) {
        messages_.Say(where, "%s"_warn_en_US, text);
      }
    }
    if (r.outcome != Outcome::Ok) {
      // The other clauses are still checked, as in mmc, for the modes
      // that their calls infer, but not reported.
      if (r.outcome == Outcome::Error && !result_.inferable_.empty()) {
        for (const ClauseTypes *other : own.drop_front(n + 1)) {
          ClauseModes ignored{nullptr, &procedure, {}, {}};
          CheckClause(own.size() > 1 ? WithHeadUnifications(*other) : *other,
              procedure, ignored);
        }
        warnings_.clear();
      }
      break;
    }
    checked.push_back(modes);
  }
  if (r.outcome == Outcome::Ok && !checked.empty()) {
    r = CheckClauses(checked);
  }
  if (r.outcome == Outcome::Error && procedure.inferred) {
    // mmc rejects the mode (REJECTED :- mode p(out(...)), with
    // --verbose-error-messages), and reports its error, at the
    // predicate's first clause, or one in a lambda expression at it, only
    // for a call that no other mode can run.
    llvm::StringRef error{r.text};
    error.consume_front(InClause() + ": ");
    rejections_[&procedure] = {r.lambda ? r.where : Start(own.front()->clause->tokens),
        "in the clause for the inferred mode " + RejectedText(procedure) +
            ": " + error.str()};
  } else if (r.outcome == Outcome::Error) {
    // Only the first mode error of a procedure, as in mmc.
    auto &message{messages_.Say(r.where, "%s"_err_en_US, r.text)};
    for (const auto &[where, text] : r.notes) {
      message.Attach(where, "%s"_because_en_US, text);
    }
  }
  return r;
}

// A rejected inferred mode as mmc writes it, with its final insts as
// `...`: p((ground >> ...), out(...)).
std::string ProcedureChecker::RejectedText(const Procedure &procedure) {
  std::string text{procedure.symbol->name.str()};
  if (!procedure.modes.empty()) {
    text += '(';
    llvm::interleave(
        procedure.modes,
        [&](const ArgMode &mode) {
          text += ops_.IsFree(mode.initial)
              ? std::string{"out(...)"}
              : "(" + InstText(mode.initial) + " >> ...)";
        },
        [&] { text += ", "; });
    text += ')';
  }
  return text;
}

// "in the clause for p(in, out)", or for an instance's method.
std::string ProcedureChecker::InClause() const {
  // mmc: "In clause for type class method implementation for `m(in,
  // out)'".
  return "in the clause for " +
      (method_ ? "the method " + ProcedureToString(*procedure_) +
                  " of instance " + method_->instanceText
               : ProcedureToString(*procedure_));
}

ProcedureChecker::Result ProcedureChecker::CheckClause(
    const ClauseTypes &clause, const Procedure &procedure, ClauseModes &out) {
  clause_ = &clause;
  form_ = clause.clause;
  procedure_ = &procedure;
  goals_.assign(form_->goals, GoalModes{});
  befores_.assign(form_->goals, {});
  touched_.assign(form_->vars.size(), 0);
  locks_.assign(form_->vars.size(), Lock::None);
  last_ = Start(form_->tokens);
  InstMap map(form_->vars.size(), insts_.Free());
  Live live(form_->vars.size(), false);
  for (std::size_t i{0}; i < form_->head.size() && i < procedure.modes.size();
       ++i) {
    map[form_->head[i]] = procedure.modes[i].initial;
    live[form_->head[i]] = HeadLive(procedure, i);
  }
  Result r{Check(form_->body, map, live)};
  if (r.outcome != Outcome::Ok) {
    return r;
  }
  std::vector<const Inst *> final;
  for (VarId var : form_->head) {
    final.push_back(map[var]);
  }
  out.final = result_.arena_.SaveArray(llvm::ArrayRef<const Inst *>{final});
  for (std::size_t i{0}; i < goals_.size(); ++i) {
    goals_[i].before =
        result_.arena_.SaveArray(llvm::ArrayRef<const Inst *>{befores_[i]});
  }
  out.goals = result_.arena_.SaveArray(llvm::ArrayRef<GoalModes>{goals_});
  return Ok();
}

// A clause of a predicate with several, as mmc checks it: with new head
// variables, HeadVar__N, and a body that unifies each with the clause's
// argument, at the argument, and then runs the clause's body. The clause's
// arguments are then free at first, and used only until unified.
const ClauseTypes &ProcedureChecker::WithHeadUnifications(
    const ClauseTypes &clause) {
  common::Arena &arena{result_.arena_};
  const ClauseForm &form{*clause.clause};
  std::vector<VarInfo> vars{form.vars.begin(), form.vars.end()};
  std::vector<const Type *> types{clause.vars.begin(), clause.vars.end()};
  std::vector<llvm::ArrayRef<VarId>> nonlocals{
      form.nonlocals.begin(), form.nonlocals.end()};
  std::vector<VarId> head;
  std::vector<Goal> goals;
  for (VarId argument : form.head) {
    VarId var{static_cast<VarId>(vars.size())};
    TokenRange tokens{form.vars[argument].tokens};
    vars.push_back({VarKind::Fresh, {}, tokens});
    types.push_back(clause.vars[argument]);
    head.push_back(var);
    const Unify &unify{result_.unifies_.emplace_back(
        Unify{tokens, static_cast<unsigned>(nonlocals.size()), var, true,
            argument, Functor{}, {}})};
    heads_.insert(&unify);
    VarId both[]{var, argument};
    nonlocals.push_back(arena.SaveArray(llvm::ArrayRef<VarId>{both}));
    goals.push_back(&unify);
  }
  goals.push_back(form.body);
  const Conj &body{result_.conjs_.emplace_back(
      Conj{form.tokens, static_cast<unsigned>(nonlocals.size()), false,
          arena.SaveArray(llvm::ArrayRef<Goal>{goals})})};
  nonlocals.push_back(arena.SaveArray(llvm::ArrayRef<VarId>{head}));
  ClauseForm &copy{result_.forms_.emplace_back(form)};
  copy.head = arena.SaveArray(llvm::ArrayRef<VarId>{head});
  copy.body = &body;
  copy.vars = arena.SaveArray(llvm::ArrayRef<VarInfo>{vars});
  copy.goals = nonlocals.size();
  copy.nonlocals = arena.SaveArray(
      llvm::ArrayRef<llvm::ArrayRef<VarId>>{nonlocals});
  std::vector<const Symbol *> symbols{
      clause.symbols.begin(), clause.symbols.end()};
  symbols.resize(copy.goals, nullptr);
  std::vector<const parser::SymName *> names{
      clause.names.begin(), clause.names.end()};
  names.resize(copy.goals, nullptr);
  ClauseTypes &typed{result_.clauseTypes_.emplace_back(clause)};
  typed.clause = &copy;
  typed.vars = arena.SaveArray(llvm::ArrayRef<const Type *>{types});
  typed.symbols = arena.SaveArray(llvm::ArrayRef<const Symbol *>{symbols});
  typed.names =
      arena.SaveArray(llvm::ArrayRef<const parser::SymName *>{names});
  return typed;
}

// A head variable is used after the clause, unless its final inst is
// clobbered, as for di (Overton, 3.2.3).
bool ProcedureChecker::HeadLive(const Procedure &procedure, std::size_t i) const {
  return LiveAfter(procedure.modes[i]);
}

bool ProcedureChecker::LiveAfter(const ArgMode &mode) const {
  const Inst *final{mode.final};
  while (final->kind == InstKind::Defined) {
    final = insts_.Expand(final);
  }
  return !((final->kind == InstKind::Ground || final->kind == InstKind::Any) &&
      (final->uniqueness == Uniqueness::Clobbered ||
          final->uniqueness == Uniqueness::MostlyClobbered));
}

parser::CharBlock ProcedureChecker::HeadEnd(const ClauseForm &form) const {
  int depth{0};
  for (std::size_t i{form.tokens.first}; i <= form.tokens.last; ++i) {
    std::string spelling{tokens_.tokens[i].spelling.ToString()};
    if (spelling == "(" || spelling == "[" || spelling == "{") {
      ++depth;
    } else if (spelling == ")" || spelling == "]" || spelling == "}") {
      --depth;
    } else if (depth == 0 && (spelling == ":-" || spelling == "-->")) {
      return tokens_.tokens[i].spelling;
    }
  }
  return Start(form.tokens);
}

// A predicate's clauses are the branches of a disjunction, as in mmc: the
// head's insts at the end of each are merged, and must match the final
// insts of the procedure's modes. The clauses were checked in order, so
// the clause, and the goal run last, are the last clause's.
ProcedureChecker::Result ProcedureChecker::CheckClauses(
    llvm::ArrayRef<ClauseModes> clauses) {
  if (clauses.size() == 1) {
    return CheckFinal(clauses.front().final, false);
  }
  std::vector<const Inst *> merged;
  std::optional<Result> error;
  for (std::size_t i{0}; i < form_->head.size(); ++i) {
    const Type *type{TypeOf(form_->head[i])};
    const Inst *inst{insts_.NotReached()};
    for (const ClauseModes &clause : clauses) {
      inst = inst ? ops_.Merge(inst, clause.final[i], type) : nullptr;
    }
    if (!inst) {
      std::vector<const Inst *> insts;
      std::vector<parser::CharBlock> branches;
      for (const ClauseModes &clause : clauses) {
        insts.push_back(clause.final[i]);
        branches.push_back(HeadEnd(*clause.clause->clause));
      }
      if ((i < procedure_->modes.size() && HeadLive(*procedure_, i)) ||
          llvm::none_of(
              insts, [&](const Inst *inst) { return ops_.IsFree(inst); })) {
        // Each such argument, as in a disjunction.
        Result r{Mismatch("disjunction", "HeadVar__" + std::to_string(i + 1),
            insts, branches, type, !error)};
        if (!error) {
          error = std::move(r);
        } else {
          llvm::append_range(error->notes, r.notes);
        }
        continue;
      }
      inst = clauses.back().final[i];
    }
    merged.push_back(inst);
  }
  if (error) {
    return *error;
  }
  return CheckFinal(merged, true);
}

// The head's insts at the end must match the final insts of the
// procedure's modes.
ProcedureChecker::Result ProcedureChecker::CheckFinal(
    llvm::ArrayRef<const Inst *> final, bool clauses) {
  if (procedure_->inferred) {
    // An inferred procedure's final insts are those of its clauses,
    // normalised.
    inferredFinal_.clear();
    for (std::size_t i{0}; i < form_->head.size() && i < final.size(); ++i) {
      inferredFinal_.push_back(ops_.Normalise(final[i], TypeOf(form_->head[i])));
    }
    return Ok();
  }
  for (std::size_t i{0}; i < form_->head.size(); ++i) {
    VarId var{form_->head[i]};
    if (i >= procedure_->modes.size()) {
      continue;
    }
    const Inst *expected{procedure_->modes[i].final};
    const Type *type{TypeOf(var)};
    if (ops_.Matches(final[i], expected, type)) {
      continue;
    }
    return {Outcome::Error, last_,
        InClause() + ": " +
            FinalError("argument " + std::to_string(i + 1) + ", " +
                    (clauses ? "HeadVar__" + std::to_string(i + 1) : Name(var)) +
                    ",",
                final[i], expected, type)};
  }
  return Ok();
}

std::string ProcedureChecker::FinalError(const std::string &subject,
    const Inst *final, const Inst *expected, const Type *type) {
  // mmc: "argument 1 did not get sufficiently instantiated", "became too
  // instantiated", or "had the wrong instantiatedness", with "Final
  // instantiatedness of `X' was `free', expected final instantiatedness
  // was `ground'".
  // An expected inst with inst variables counts as one with free parts,
  // as in mmc.
  bool parameters{false};
  insts_.Substitute(expected, [&](const Inst &parameter) {
    parameters = true;
    return &parameter;
  });
  // And one that the expected inst matches, as ground for unique, as
  // one not instantiated enough.
  const char *how{ops_.HasFree(final, type) ? "did not get sufficiently instantiated"
          : parameters                     ? "became too instantiated"
          : ops_.Matches(expected, final, type)
          ? "did not get sufficiently instantiated"
          : ops_.HasFree(expected, type) ? "became too instantiated"
                                         : "had the wrong inst"};
  return "mode error: " + subject + " " + how + ": its final inst is " +
      InstText(final) + ", but " + Expected(expected, final) + " is expected";
}

std::string ProcedureChecker::Expected(
    const Inst *expected, const Inst *actual) {
  if (expected->kind == InstKind::Ground &&
      actual->kind == InstKind::HigherOrder &&
      actual->predOrFunc == parser::PredOrFunc::Function &&
      !actual->modes.empty()) {
    return InstText(ops_.DefaultFunction(actual->modes.size() - 1));
  }
  return InstText(expected);
}

void ProcedureChecker::Flatten(const Goal &goal, std::vector<Goal> &out) {
  if (const auto *conj{std::get_if<const Conj *>(&goal)};
      conj && !(*conj)->parallel) {
    for (const Goal &g : (*conj)->goals) {
      Flatten(g, out);
    }
  } else {
    out.push_back(goal);
  }
}

// Checks a goal; a unification or call that binds a locked variable that
// is free is a scope error, which waits for the variable to be bound
// outside.
ProcedureChecker::Result ProcedureChecker::Check(
    const Goal &goal, InstMap &map, const Live &live) {
  Result r{CheckLocked(goal, map, live)};
  if (r.outcome == Outcome::Ok &&
      llvm::any_of(
          map, [&](const Inst *inst) { return ops_.IsNotReached(inst); })) {
    ++unreachables_;
  }
  return r;
}

ProcedureChecker::Result ProcedureChecker::CheckLocked(
    const Goal &goal, InstMap &map, const Live &live) {
  if ((!std::holds_alternative<const Unify *>(goal) &&
          !std::holds_alternative<const Call *>(goal)) ||
      llvm::all_of(locks_, [](Lock lock) { return lock == Lock::None; })) {
    return CheckGoal(goal, map, live);
  }
  llvm::ArrayRef<VarId> vars{form_->nonlocals[NumberOf(goal)]};
  std::vector<const Inst *> before;
  for (VarId var : vars) {
    before.push_back(map[var]);
  }
  Result r{CheckGoal(goal, map, live)};
  if (r.outcome != Outcome::Ok) {
    return r;
  }
  for (std::size_t i{0}; i < vars.size(); ++i) {
    VarId var{vars[i]};
    // Only a free variable, as in mmc: a goal may bind the free parts of
    // a partially instantiated one.
    if (locks_[var] == Lock::None || !ops_.IsFree(before[i]) ||
        ops_.IsFree(map[var]) || ops_.IsNotReached(map[var])) {
      continue;
    }
    // mmc: "scope error: attempt to bind a non-local variable inside a
    // negation. Variable `X' has instantiatedness `free', expected
    // instantiatedness was `unique(1)'."
    return Error(goal,
        "scope error: " + Name(var) + " is bound inside " +
            (locks_[var] == Lock::Negation
                    ? std::string{"a negation, but is not local to it"}
                    : std::string{"the condition of an if-then-else, but "
                                  "occurs outside the condition and the then "
                                  "part"}) +
            ": it has inst " + InstText(before[i]) + ", and would get " +
            InstText(map[var]),
        {var});
  }
  return r;
}

ProcedureChecker::Result ProcedureChecker::CheckGoal(
    const Goal &goal, InstMap &map, const Live &live) {
  // The insts before the goal, for determinism analysis; the run that
  // succeeds is the last.
  std::vector<const Inst *> &before{befores_[NumberOf(goal)]};
  before.clear();
  for (VarId var : form_->nonlocals[NumberOf(goal)]) {
    before.push_back(map[var]);
  }
  return common::visit(
      common::visitors{
          [&](const Conj *conj) -> Result {
            if (conj->parallel) {
              return Unsupported(Unchecked::ParallelConjunction);
            }
            // true counts as a goal run, as in mmc, except as the body of
            // a clause, and in a conjunction, which drops it.
            if (conj->goals.empty() && goal != form_->body) {
              last_ = Start(conj->tokens);
            }
            std::vector<Goal> goals;
            Flatten(goal, goals);
            return Conjunction(goal, goals, map, live);
          },
          [&](const Disj *disj) { return Disjunction(goal, *disj, map, live); },
          [&](const IfThenElse *ite) {
            return IfThenElseGoal(goal, *ite, map, live);
          },
          [&](const Not *negation) {
            return Negation(goal, *negation, map, live);
          },
          [&](const Unify *unify) { return CheckUnify(*unify, goal, map, live); },
          [&](const Call *call) -> Result {
            const Symbol *callee{clause_->symbols[call->number]};
            if (!callee || callee->kind != SymbolKind::Predicate ||
                callee->arity != call->arguments.size()) {
              return Unsupported(Unchecked::TypeErrors);
            }
            Result r{CheckCall(goal, *callee, call->arguments, false, map,
                goals_[call->number], live)};
            if (r.outcome == Outcome::Ok) {
              goals_[call->number].unify = UnifyKind::Call;
              last_ = Start(call->tokens);
            }
            return r;
          },
          [&](const HOCall *call) {
            return CheckHigherOrderCall(goal, *call, map, live);
          },
          [&](const Scope *scope) -> Result {
            switch (scope->kind) {
            case ScopeKind::Trace:
              return Unsupported(Unchecked::Scope);
            case ScopeKind::PromisePurity: {
              ++promised_;
              Result r{Check(scope->goal, map, live)};
              --promised_;
              return r;
            }
            default:
              // The others change nothing about modes.
              return Check(scope->goal, map, live);
            }
          },
          [&](const Try *) { return Unsupported(Unchecked::Try); },
          [&](const Atomic *) { return Unsupported(Unchecked::Atomic); },
      },
      goal);
}

// Each branch of a disjunction starts from the insts before it, and their
// results are merged (Overton, 3.1.4). fail, the empty disjunction, is not
// reached after.
ProcedureChecker::Result ProcedureChecker::Disjunction(
    const Goal &goal, const Disj &disj, InstMap &map, const Live &live) {
  if (disj.goals.empty()) {
    last_ = Start(disj.tokens);
    map.assign(map.size(), insts_.NotReached());
    return Ok();
  }
  std::vector<InstMap> maps;
  std::vector<parser::CharBlock> branches;
  for (const Goal &branch : disj.goals) {
    InstMap inner{map};
    Result r{Check(branch, inner, live)};
    if (r.outcome != Outcome::Ok) {
      return r;
    }
    maps.push_back(std::move(inner));
    branches.push_back(Start(TokensOf(branch)));
  }
  return Merge(goal, "disjunction", maps, branches, map, live);
}

// The condition may bind only what is local to it and the then part, which
// starts from its insts; the else part starts from the insts before the
// if-then-else, and the two are merged.
ProcedureChecker::Result ProcedureChecker::IfThenElseGoal(
    const Goal &goal, const IfThenElse &ite, InstMap &map, const Live &live) {
  std::vector<Lock> locks{locks_};
  for (VarId var : form_->nonlocals[ite.number]) {
    if (locks_[var] == Lock::None) {
      locks_[var] = Lock::Condition;
    }
  }
  Live after{live};
  for (VarId var : form_->nonlocals[NumberOf(ite.then)]) {
    after[var] = true;
  }
  InstMap then{map};
  Result r{Check(ite.condition, then, after)};
  locks_ = std::move(locks);
  if (r.outcome != Outcome::Ok) {
    return r;
  }
  llvm::ArrayRef<VarId> condition{form_->nonlocals[NumberOf(ite.condition)]};
  std::vector<VarId> vars;
  for (VarId var : form_->nonlocals[ite.number]) {
    if (llvm::is_contained(condition, var)) {
      vars.push_back(var);
    }
  }
  if (std::optional<Result> error{AnyInNegatedContext(
          goal, vars, map, "if-then-else", "condition")}) {
    return *error;
  }
  r = Check(ite.then, then, live);
  if (r.outcome != Outcome::Ok) {
    return r;
  }
  InstMap otherwise{map};
  r = Check(ite.otherwise, otherwise, live);
  if (r.outcome != Outcome::Ok) {
    return r;
  }
  return Merge(goal, "if-then-else", {then, otherwise},
      {Start(TokensOf(ite.then)), Start(TokensOf(ite.otherwise))}, map, live);
}

// A negation may bind only what is local to it, and binds nothing after.
ProcedureChecker::Result ProcedureChecker::Negation(const Goal &goal,
    const Not &negation, InstMap &map, const Live &live) {
  std::vector<Lock> locks{locks_};
  for (VarId var : form_->nonlocals[negation.number]) {
    if (locks_[var] == Lock::None) {
      locks_[var] = Lock::Negation;
    }
  }
  // Nothing after it is run if the negated goal succeeds, so nothing is
  // used after that goal but by its own goals, as in mmc, which takes it
  // as `if G then fail else true`; the values used after the negation are
  // needed on backtracking, which uniqueness checking sees to.
  InstMap inner{map};
  Result r{Check(negation.goal, inner, Live(live.size(), false))};
  locks_ = std::move(locks);
  if (r.outcome == Outcome::Ok) {
    if (std::optional<Result> error{AnyInNegatedContext(goal,
            form_->nonlocals[negation.number], map, "negation", "body")}) {
      return *error;
    }
    // mmc takes the negation of true, or of a conjunction of trues, as
    // fail, and so nothing after it is reached.
    std::vector<Goal> goals;
    Flatten(negation.goal, goals);
    if (goals.empty()) {
      map.assign(map.size(), insts_.NotReached());
    }
  }
  return r;
}

// Negation as failure is unsound if a negated goal may constrain a
// solver variable, so a nonlocal of a negation or of a condition whose
// inst has an any part needs a purity promise around the goal (reference
// manual, "Solver types and negated contexts"). The goal's own errors come
// first, as in mmc, whose promise_pure pragma and impure declarations do
// not count; neither does an any_pred closure (see the register).
// mmc: "purity error: negation should be inside a promise_purity scope
// because non-local variable `X' has inst `any' and appears in the body."
std::optional<ProcedureChecker::Result> ProcedureChecker::AnyInNegatedContext(
    const Goal &goal, llvm::ArrayRef<VarId> vars, const InstMap &map,
    const char *context, const char *part) {
  if (promised_ > 0) {
    return std::nullopt;
  }
  for (VarId var : vars) {
    if (ops_.HasAny(map[var], TypeOf(var))) {
      return Error(goal,
          std::string{"purity error: the "} + context +
              " must be inside a promise_purity scope, because nonlocal " +
              Name(var) + " has inst any and appears in its " + part);
    }
  }
  return std::nullopt;
}

// The branches' insts of the goal's nonlocals merged into `map`. A
// variable whose insts have no merge, because it is free in some branches
// and bound in others, or a part of it is, is an error, reported at the
// goal run last, with a note at each branch reached, as in mmc; but not a
// variable that is free in some branch and not used after. Each such
// variable is reported, as in mmc, those after the first in notes.
ProcedureChecker::Result ProcedureChecker::Merge(const Goal &goal,
    const char *what, llvm::ArrayRef<InstMap> maps,
    llvm::ArrayRef<parser::CharBlock> branches, InstMap &map,
    const Live &live) {
  auto reached{[&](const InstMap &m) {
    return llvm::none_of(
        m, [&](const Inst *inst) { return ops_.IsNotReached(inst); });
  }};
  std::optional<Result> error;
  for (VarId var : form_->nonlocals[NumberOf(goal)]) {
    const Type *type{TypeOf(var)};
    const Inst *merged{insts_.NotReached()};
    for (const InstMap &m : maps) {
      merged = merged ? ops_.Merge(merged, m[var], type) : nullptr;
    }
    if (!merged) {
      std::vector<const Inst *> insts;
      for (const InstMap &m : maps) {
        insts.push_back(m[var]);
      }
      if (live[var] || llvm::none_of(insts, [&](const Inst *inst) {
            return ops_.IsFree(inst);
          })) {
        // It waits for that variable, as in mmc.
        Result r{Mismatch(what, Name(var), insts, branches, type, !error)};
        if (!error) {
          error = std::move(r);
          error->waits = {var};
        } else {
          error->waits.push_back(var);
          llvm::append_range(error->notes, r.notes);
        }
        continue;
      }
      merged = maps.front()[var];
    }
    map[var] = merged;
  }
  if (error) {
    return *error;
  }
  if (llvm::none_of(maps, reached)) {
    map.assign(map.size(), insts_.NotReached());
  }
  return Ok();
}

// mmc: "mode mismatch in disjunction. The variable `X' is ground in some
// branches but not others. In this branch, `X' is ground. In this branch,
// `X' is free." With three or more branches, where it is free in those
// where it is not ground: "The variable `X' is ground in 2 out of 3
// branches. In this branch, `X' is free."
ProcedureChecker::Result ProcedureChecker::Mismatch(const char *what,
    const std::string &name, llvm::ArrayRef<const Inst *> insts,
    llvm::ArrayRef<parser::CharBlock> branches, const Type *type, bool first) {
  bool ground{llvm::any_of(
      insts, [&](const Inst *inst) { return ops_.IsGround(inst, type); })};
  std::string mismatch{name +
      (ground ? " is ground in some branches but not others"
              : " has insts in its branches that do not merge")};
  Result r{Outcome::Error, last_,
      InClause() + ": mode mismatch in the " + what + ": " + mismatch};
  if (!first) {
    // A variable after the first, in notes of the first's error.
    r.notes.emplace_back(last_, "and " + mismatch);
  }
  // Of three or more branches, counting those not reached, where it is
  // ground in all but one, and free in that one, only that one has a note,
  // as in mmc ("ground in 2 out of 3 branches"). A branch not reached
  // never has one.
  bool brief{insts.size() >= 3 &&
      llvm::count_if(insts, [&](const Inst *inst) {
        return !ops_.IsGround(inst, type);
      }) == 1 &&
      llvm::any_of(insts, [&](const Inst *inst) { return ops_.IsFree(inst); })};
  for (std::size_t i{0}; i < insts.size(); ++i) {
    if (!ops_.IsNotReached(insts[i]) &&
        (!brief || !ops_.IsGround(insts[i], type))) {
      r.notes.emplace_back(branches[i],
          "in this branch, " + name + " has inst " + InstText(insts[i]));
    }
  }
  return r;
}

// The goals of a conjunction run in source order, except that one that
// would be a mode error waits. When a goal runs, the waiting goals whose
// variables it changed are woken: they run next, in the order they
// waited, before the other goals woken earlier, and one that still cannot
// run waits again, after the others (Overton, 3.4). If goals are left
// waiting, the error of the first is reported, as in mmc.
ProcedureChecker::Result ProcedureChecker::Conjunction(const Goal &conj,
    llvm::ArrayRef<Goal> goals, InstMap &map, const Live &live) {
  struct Waiting {
    Goal goal;
    std::vector<const Inst *> insts; // of its variables, when it waited
    Result error;
  };
  std::vector<Goal> order;
  std::vector<Waiting> waiting;
  std::deque<Goal> agenda; // those woken, to run next
  std::size_t next{0}; // the next goal in source order
  // The goals, by number, that wait for no variable and that a goal with
  // insts not reached woke; they are not retried at the end, as in mmc.
  std::set<unsigned> kept;
  // The insts of the variables a goal waits for, and how often each was
  // put in a term free; mmc wakes a goal only when these change.
  auto snapshot{[&](const Goal &goal, const Result &error) {
    std::vector<const Inst *> insts;
    if (error.never) {
      return insts;
    }
    llvm::ArrayRef<VarId> vars{error.waits};
    if (vars.empty()) {
      vars = form_->nonlocals[NumberOf(goal)];
    }
    for (VarId var : vars) {
      insts.push_back(map[var]);
      insts.push_back(reinterpret_cast<const Inst *>(
          static_cast<std::uintptr_t>(touched_[var])));
    }
    return insts;
  }};
  // Runs a goal if it can; its error if not.
  auto run{[&](const Goal &goal) -> Result {
    // What the goals not run yet use is live after this one.
    Live after{live};
    auto use{[&](const Goal &other) {
      if (NumberOf(other) != NumberOf(goal)) {
        for (VarId var : form_->nonlocals[NumberOf(other)]) {
          after[var] = true;
        }
      }
    }};
    for (std::size_t i{next}; i < goals.size(); ++i) {
      use(goals[i]);
    }
    for (const Waiting &w : waiting) {
      use(w.goal);
    }
    for (const Goal &w : agenda) {
      use(w);
    }
    InstMap trial{map};
    parser::CharBlock last{last_};
    std::size_t warnings{warnings_.size()};
    std::vector<unsigned> touched{touched_};
    unsigned unreachables{unreachables_};
    Result r{Check(goal, trial, after)};
    if (r.outcome != Outcome::Ok) {
      last_ = last;
      warnings_.resize(warnings);
      touched_ = std::move(touched);
      unreachables_ = unreachables;
      return r;
    }
    map = std::move(trial);
    order.push_back(goal);
    // After a goal that cannot succeed, nothing is reached, and every
    // goal's mode is right (Overton, 3.1.4).
    if (llvm::any_of(
            map, [&](const Inst *inst) { return ops_.IsNotReached(inst); })) {
      map.assign(map.size(), insts_.NotReached());
    }
    return r;
  }};
  // Puts the waiting goals whose variables changed first on the agenda.
  auto wake{[&] {
    std::vector<Waiting> still;
    std::vector<Goal> woken;
    for (Waiting &w : waiting) {
      if (snapshot(w.goal, w.error) == w.insts) {
        still.push_back(std::move(w));
      } else {
        woken.push_back(w.goal);
      }
    }
    waiting = std::move(still);
    agenda.insert(agenda.begin(), woken.begin(), woken.end());
  }};
  // After a goal in which some insts were not reached, as in a branch,
  // condition or negation that cannot succeed, every waiting goal is woken,
  // as if each of its variables had changed, as in mmc: in the order of
  // the first variable each waits for, those that wait for none last.
  auto wakeAll{[&] {
    auto first{[&](const Waiting &w) {
      llvm::ArrayRef<VarId> vars{w.error.waits};
      if (vars.empty() && !w.error.never) {
        vars = form_->nonlocals[NumberOf(w.goal)];
      }
      return vars.empty() ? std::numeric_limits<VarId>::max()
                          : *std::min_element(vars.begin(), vars.end());
    }};
    std::stable_sort(waiting.begin(), waiting.end(),
        [&](const Waiting &a, const Waiting &b) { return first(a) < first(b); });
    std::vector<Goal> woken;
    for (const Waiting &w : waiting) {
      woken.push_back(w.goal);
      if (w.error.never) {
        kept.insert(NumberOf(w.goal));
      }
    }
    waiting.clear();
    agenda.insert(agenda.begin(), woken.begin(), woken.end());
  }};
  // When no goal is left to run, the waiting goals are all retried once,
  // since goals that ran may have left their variables unused, as in mmc.
  bool progress{false};
  while (true) {
    Goal goal;
    if (!agenda.empty()) {
      goal = agenda.front();
      agenda.pop_front();
    } else if (next < goals.size()) {
      goal = goals[next++];
    } else if (progress && !waiting.empty()) {
      progress = false;
      std::vector<Waiting> still;
      for (Waiting &w : waiting) {
        if (kept.count(NumberOf(w.goal))) {
          still.push_back(std::move(w));
        } else {
          agenda.push_back(w.goal);
        }
      }
      waiting = std::move(still);
      continue;
    } else {
      break;
    }
    unsigned unreachables{unreachables_};
    Result r{run(goal)};
    if (r.outcome == Outcome::Unsupported) {
      return r;
    }
    if (r.outcome == Outcome::Error) {
      std::vector<const Inst *> insts{snapshot(goal, r)};
      waiting.push_back({goal, std::move(insts), std::move(r)});
      continue;
    }
    progress = true;
    // After a goal that cannot succeed, the others are not checked, and
    // the goal is the last run, as in mmc.
    if (!map.empty() && ops_.IsNotReached(map.front())) {
      waiting.clear();
      break;
    }
    if (unreachables_ != unreachables) {
      wakeAll();
    } else {
      wake();
    }
  }
  if (!waiting.empty()) {
    // A head unification waits until the goals do, but its error is
    // reported only if no other goal waits, as in mmc.
    auto first{llvm::find_if(waiting, [&](const Waiting &w) {
      const auto *unify{std::get_if<const Unify *>(&w.goal)};
      return !unify || !heads_.count(*unify);
    })};
    Result r{(first != waiting.end() ? *first : waiting.front()).error};
    // The conjunction, as a goal of another, waits for what any of its
    // goals waits for.
    std::vector<VarId> waits;
    for (const Waiting &w : waiting) {
      llvm::ArrayRef<VarId> vars{w.error.waits};
      if (w.error.never) {
        continue;
      }
      if (vars.empty()) {
        vars = form_->nonlocals[NumberOf(w.goal)];
      }
      for (VarId var : vars) {
        if (!llvm::is_contained(waits, var)) {
          waits.push_back(var);
        }
      }
    }
    r.waits = std::move(waits);
    r.never = llvm::all_of(waiting, [](const Waiting &w) { return w.error.never; });
    return r;
  }
  goals_[NumberOf(conj)].order =
      result_.arena_.SaveArray(llvm::ArrayRef<Goal>{order});
  return Ok();
}

ProcedureChecker::Result ProcedureChecker::CheckVarUnify(
    const Unify &unify, const Goal &goal, InstMap &map, const Live &live) {
  VarId x{unify.left}, y{unify.var};
  const Type *type{TypeOf(x)};
  const Inst *a{map[x]}, *b{map[y]};
  GoalModes &modes{goals_[unify.number]};
  modes.head = heads_.count(&unify);
  bool xFree{ops_.IsFree(a)}, yFree{ops_.IsFree(b)};
  if (xFree && yFree) {
    // Two free variables may be aliased only if one is not used again.
    // mmc: "mode error in unification of `X' and `Y'. Variable `X' has
    // instantiatedness `free', variable `Y' has instantiatedness `free'."
    if (live[x] && live[y]) {
      return Error(goal,
          "mode error in the unification of " + Name(x) + " and " + Name(y) +
              ", which are both free",
          {x, y});
    }
    modes.unify = UnifyKind::Assign;
    modes.assigned = live[x] ? x : y;
    modes.dead = true;
    // That wakes the goals that wait for either, which may now find it
    // not used again, as in mmc.
    ++touched_[x];
    ++touched_[y];
    return Ok();
  }
  if (xFree || yFree) {
    VarId to{xFree ? x : y}, from{xFree ? y : x};
    // A term with free parts would alias them, unless one side is not
    // used again. mmc: "mode error in unification of `V13' and `V1'.
    // Variable `V13' has instantiatedness `free', variable `V1' has
    // instantiatedness `unique('[|]'(free, free))'."
    if (live[x] && live[y] && ops_.HasFree(map[from], type)) {
      return Error(goal,
          "mode error in the unification of " + Name(x) + " and " + Name(y) +
              ": " + Name(x) + " has inst " + InstText(a) + ", and " +
              Name(y) + " has inst " + InstText(b),
          {x, y});
    }
    // Both stay; a value used by both is shared.
    const Inst *value{live[from] ? insts_.Share(map[from]) : map[from]};
    map[to] = value;
    map[from] = value;
    modes.unify = UnifyKind::Assign;
    modes.assigned = to;
    modes.dead = !live[to];
    return Ok();
  }
  if (type && type->kind == TypeKind::HigherOrder) {
    return HigherOrderUnification(goal, x, Name(y));
  }
  const Inst *unified{ops_.Unify(a, b, type)};
  if (!unified || (ops_.HasFree(a, type) && ops_.HasFree(b, type))) {
    return Error(goal,
        "mode error in the unification of " + Name(x) + " and " + Name(y) +
            ": " + Name(x) + " has inst " + InstText(a) + ", and " + Name(y) +
            " has inst " + InstText(b),
        {x, y});
  }
  if (ops_.IsNotReached(unified) && !ops_.IsNotReached(a) &&
      !ops_.IsNotReached(b)) {
    // mmc: "warning: unification of `X' and `Y' cannot succeed".
    modes.fails = true;
    warnings_.emplace_back(Source(unify.tokens),
        "the unification of " + Name(x) + " and " + Name(y) +
            " cannot succeed: " + Name(x) + " has inst " + InstText(a) +
            ", and " + Name(y) + " has inst " + InstText(b));
  }
  // Two references used after are shared (Overton, definition 3.2.19).
  if (live[x] && live[y]) {
    unified = insts_.Share(unified);
  }
  map[x] = unified;
  map[y] = unified;
  modes.unify = UnifyKind::Test;
  return Ok();
}

ProcedureChecker::Result ProcedureChecker::CheckUnify(
    const Unify &unify, const Goal &goal, InstMap &map, const Live &live) {
  Result r{unify.isVar ? CheckVarUnify(unify, goal, map, live) : Ok()};
  if (unify.isVar) {
    if (r.outcome == Outcome::Ok) {
      last_ = Start(unify.tokens);
    }
    return r;
  }
  GoalModes &modes{goals_[unify.number]};
  VarId x{unify.left};
  const Type *type{TypeOf(x)};
  InstFunctor functor{{}, {}, static_cast<unsigned>(unify.arguments.size()),
      false};
  const Symbol *symbol{clause_->symbols[unify.number]};
  switch (unify.functor.kind) {
  case FunctorKind::Lambda:
    r = CheckLambda(unify, goal, map, live);
    modes.dead = !live[x];
    return r;
  case FunctorKind::Coerce:
    return CheckCoerce(unify, goal, map, live);
  case FunctorKind::Literal:
    functor.name = unify.functor.literal->text;
    functor.literal = true;
    break;
  case FunctorKind::ImplementationLiteral:
    functor.name = unify.functor.implementation->name;
    functor.literal = true;
    break;
  case FunctorKind::Tuple:
    functor.name = "{}";
    break;
  default:
    if (!symbol) {
      // A character constant.
      functor.name = unify.functor.name->name;
      break;
    }
    if ((symbol->kind == SymbolKind::Function &&
            unify.arguments.size() < symbol->arity) ||
        (symbol->kind == SymbolKind::Predicate &&
            unify.arguments.size() <= symbol->arity)) {
      r = CheckClosure(unify, goal, *symbol, map);
      modes.dead = !live[x];
      return r;
    }
    if (symbol->kind == SymbolKind::Function ||
        symbol->kind == SymbolKind::Field ||
        symbol->kind == SymbolKind::FieldUpdate) {
      if (symbol->arity != unify.arguments.size()) {
        return Unsupported(Unchecked::FieldClosure);
      }
      std::vector<VarId> arguments{
          unify.arguments.begin(), unify.arguments.end()};
      arguments.push_back(x);
      r = CheckCall(goal, *symbol, arguments, true, map, modes, live);
      if (r.outcome == Outcome::Ok) {
        modes.unify = UnifyKind::Call;
        last_ = Start(unify.tokens);
      }
      return r;
    }
    if (symbol->kind != SymbolKind::Constructor ||
        symbol->arity != unify.arguments.size()) {
      return Unsupported(Unchecked::TypeErrors);
    }
    functor.module = symbol->module;
    functor.name = symbol->name;
    break;
  }
  std::vector<const Inst *> arguments;
  for (VarId var : unify.arguments) {
    arguments.push_back(map[var]);
  }
  const Inst *a{map[x]};
  if (ops_.IsFree(a)) {
    // A construction. An argument with free parts that is used again
    // would be aliased to the new term's, which mode analysis does not
    // follow, so it waits, unless the term is not used again, as in mmc.
    // The term's copy of an argument used again is shared; the argument
    // keeps its own inst, as in mmc.
    std::vector<const Inst *> shared;
    std::vector<VarId> waits;
    for (std::size_t i{0}; i < unify.arguments.size(); ++i) {
      VarId var{unify.arguments[i]};
      if (ops_.HasFree(arguments[i], TypeOf(var)) && live[var] && live[x]) {
        waits.push_back(var);
      }
      shared.push_back(live[var] ? insts_.Share(arguments[i]) : arguments[i]);
    }
    const Inst *term{insts_.Bound(Uniqueness::Unique, {{functor, shared}})};
    if (!waits.empty()) {
      // Binding X would make it a deconstruction instead.
      waits.push_back(x);
      // mmc: "mode error in unification of `L' and `list.[H | V_6]'.
      // Variable `L' has instantiatedness `free', term `list.[H | V_6]'
      // has instantiatedness `named inst list.'[|]'(free, unique([]))'."
      return Error(goal,
          "mode error in the unification of " + Name(x) + " and " +
              TermText(*form_, unify) + ": " + Name(x) + " has inst " +
              InstText(a) + ", and " + TermText(*form_, unify) +
              " has inst " + InstText(term),
          std::move(waits));
    }
    for (std::size_t i{0}; i < unify.arguments.size(); ++i) {
      // An argument with free parts is now part of the term, which wakes
      // the goals that wait for it, as in mmc.
      if (ops_.HasFree(shared[i], TypeOf(unify.arguments[i]))) {
        ++touched_[unify.arguments[i]];
      }
    }
    map[x] = term;
    modes.unify = UnifyKind::Construct;
    modes.functor = functor;
    modes.dead = !live[x];
    last_ = Start(unify.tokens);
    return Ok();
  }
  // A deconstruction, or a test if the arguments are bound too. X keeps
  // its uniqueness; a part that both X and its variable use after is
  // shared (Overton, definition 3.2.20).
  const Inst *term{insts_.Bound(Uniqueness::Unique, {{functor, arguments}})};
  const Inst *unified{ops_.Unify(a, term, type)};
  std::optional<std::vector<const Inst *>> parts;
  if (unified) {
    parts = ops_.Arguments(unified, functor, type);
  }
  std::optional<std::vector<const Inst *>> before{
      ops_.Arguments(a, functor, type)};
  bool error{!unified};
  std::vector<VarId> waits;
  for (std::size_t i{0}; before && i < arguments.size(); ++i) {
    // Free in both is an alias of two free variables.
    if (ops_.IsFree((*before)[i]) && ops_.IsFree(arguments[i]) &&
        live[unify.arguments[i]] && live[x]) {
      error = true;
      waits.push_back(unify.arguments[i]);
    }
  }
  // Binding X's arguments, or X, can make it run.
  waits.push_back(x);
  if (error) {
    return Error(goal,
        "mode error in the unification of " + Name(x) + " and " +
            TermText(*form_, unify) + ": " + Name(x) + " has inst " +
            InstText(a) + ", and " + TermText(*form_, unify) + " has inst " +
            InstText(term),
        std::move(waits));
  }
  if (ops_.IsNotReached(unified) && !ops_.IsNotReached(a)) {
    // mmc: "warning: unification of `V3' and m.c cannot succeed. `V3' has
    // instantiatedness `unique(a)'."
    modes.fails = true;
    warnings_.emplace_back(Source(unify.tokens),
        "the unification of " + Name(x) + " and " + TermText(*form_, unify) +
            " cannot succeed: " + Name(x) + " has inst " + InstText(a));
  }
  bool test{true}, sharing{false};
  for (std::size_t i{0}; i < unify.arguments.size(); ++i) {
    VarId var{unify.arguments[i]};
    test &= !ops_.IsFree(arguments[i]);
    if (parts && live[x] && live[var]) {
      const Inst *part{insts_.Share((*parts)[i])};
      sharing |= part != (*parts)[i];
      (*parts)[i] = part;
    }
    map[var] = parts ? (*parts)[i] : insts_.NotReached();
  }
  if (sharing) {
    if (const Inst *shared{ops_.Unify(unified,
            insts_.Bound(Uniqueness::Unique, {{functor, *parts}}), type)}) {
      unified = shared;
    }
  }
  map[x] = unified;
  modes.unify = test ? UnifyKind::Test : UnifyKind::Deconstruct;
  modes.functor = functor;
  last_ = Start(unify.tokens);
  return Ok();
}

llvm::ArrayRef<const Procedure *> ProcedureChecker::ProceduresFor(
    const Symbol &symbol) {
  if (symbol.kind != SymbolKind::Field &&
      symbol.kind != SymbolKind::FieldUpdate) {
    return result_.ProceduresOf(symbol);
  }
  // A field's function is in(Term) = out, and its update function
  // in(Term), in(Value) = out (reference manual, "Field access
  // functions").
  const Procedure *&procedure{fields_[&symbol]};
  if (!procedure) {
    ArgMode in{insts_.Ground(), insts_.Ground()},
        out{insts_.Free(), insts_.Ground()};
    std::vector<ArgMode> modes{in};
    if (symbol.kind == SymbolKind::FieldUpdate) {
      modes.push_back(in);
    }
    modes.push_back(out);
    procedure = &result_.imported_.emplace_back(Procedure{&symbol, 0,
        result_.arena_.SaveArray(llvm::ArrayRef<ArgMode>{modes}),
        parser::Determinism::Det, {}});
    result_.bySymbol_[&symbol].push_back(procedure);
  }
  return result_.ProceduresOf(symbol);
}

// A call runs the procedure of its callee whose initial insts the
// arguments' insts match; of several, the one whose initial insts match
// those of the others (Overton, 3.3.3).
ProcedureChecker::Result ProcedureChecker::CheckCall(const Goal &goal,
    const Symbol &callee, llvm::ArrayRef<VarId> arguments, bool function,
    InstMap &map, GoalModes &modes, const Live &live) {
  bool inferable{result_.inferable_.count(&callee) > 0};
  if (inferable &&
      llvm::any_of(arguments, [&](VarId var) { return ops_.IsNotReached(map[var]); })) {
    // Unreachable: no procedure is made for it.
    return Ok();
  }
  llvm::ArrayRef<const Procedure *> procedures{ProceduresFor(callee)};
  if (procedures.empty() && !inferable) {
    // Its missing mode declaration was reported.
    return Unsupported(Unchecked::InferredModes);
  }
  auto moreSpecific{[&](const Procedure &p, const Procedure &q) {
    for (std::size_t i{0}; i < arguments.size(); ++i) {
      if (!ops_.Matches(p.modes[i].initial, q.modes[i].initial,
              TypeOf(arguments[i]))) {
        return false;
      }
    }
    return true;
  }};
  // Each procedure's modes with its inst parameters bound by the
  // arguments (Overton, figure 4.3).
  std::map<const Procedure *, std::vector<ArgMode>> instantiated;
  for (const Procedure *procedure : procedures) {
    instantiated[procedure] = Instantiate(procedure->modes, arguments, map);
  }
  std::vector<const Procedure *> candidates;
  bool implied{false};
  for (bool allowImplied : {false, true}) {
    for (const Procedure *procedure : procedures) {
      // An inferred mode that would clobber a variable used after is not
      // that of the call, which gets one of its own, as in mmc.
      // Nor is a rejected one.
      if (procedure->modes.size() == arguments.size() &&
          !Misfit(instantiated[procedure], arguments, map, allowImplied) &&
          !(inferable &&
              (Clobbers(goal, instantiated[procedure], arguments, live,
                   [](std::size_t) { return std::string{}; }) ||
                  rejections_.count(procedure)))) {
        candidates.push_back(procedure);
      }
    }
    if (!candidates.empty()) {
      implied = allowImplied;
      break;
    }
  }
  const Procedure *chosen{nullptr};
  for (const Procedure *p : candidates) {
    if (llvm::none_of(candidates, [&](const Procedure *q) {
          return q != p && moreSpecific(*q, *p) && !moreSpecific(*p, *q);
        })) {
      chosen = p;
      break;
    }
  }
  if (inferable && !chosen) {
    // A call whose insts, normalised, are those of a rejected mode is a
    // call of it, as in mmc.
    std::vector<ArgMode> inferred{InferredModes(arguments, map, live)};
    if (auto rejected{llvm::find_if(procedures,
            [&](const Procedure *p) {
              return rejections_.count(p) &&
                  llvm::equal(p->modes, inferred,
                      [](const ArgMode &a, const ArgMode &b) {
                        return InstToString(*a.initial) ==
                            InstToString(*b.initial);
                      });
            })};
        rejected != procedures.end()) {
      return RejectedCall(goal, callee, **rejected, arguments, map);
    }
    chosen = Infer(callee, inferred);
    instantiated[chosen] = chosen->modes.vec();
    implied = false;
  }
  if (inferable) {
    auto outcome{outcomes_.find(chosen)};
    if (outcome != outcomes_.end() && outcome->second == Outcome::Unsupported) {
      // Its clauses use what mode analysis does not check yet.
      return Unsupported(Unchecked::InferredModes);
    }
  }
  std::string what{std::string{function ? "func " : "pred "} +
      callee.module.str() + '.' + callee.name.str() + '/' +
      std::to_string(callee.arity)};
  auto where{[&](std::size_t i) {
    bool result{function && i + 1 == arguments.size()};
    return (result ? std::string{"the result"}
                   : "argument " + std::to_string(i + 1)) +
        " of the call to " + what;
  }};
  if (!chosen) {
    if (procedures.size() == 1 &&
        procedures.front()->modes.size() == arguments.size()) {
      // An argument used after that the procedure would clobber is the
      // error, before one whose inst does not match, as in mmc.
      if (std::optional<Result> error{Clobbers(
              goal, instantiated[procedures.front()], arguments, live, where)}) {
        return *error;
      }
      // mmc: "in argument 1 of call to predicate `m.w'/2: mode error:
      // variable `Y' has instantiatedness `free', expected instantiatedness
      // was `ground'".
      return ArgumentError(
          goal, instantiated[procedures.front()], arguments, map, where);
    }
    // mmc: "in call to function `int.+'/2: mode error: arguments `Y, V_4'
    // and the return value `X' have the following insts: free, unique(1),
    // free which does not match any of the modes for function `int.+'/2."
    std::string names, insts;
    std::vector<VarId> waits;
    for (std::size_t i{0}; i < arguments.size(); ++i) {
      bool result{function && i + 1 == arguments.size()};
      if (ops_.IsFree(map[arguments[i]]) ||
          ops_.HasFree(map[arguments[i]], TypeOf(arguments[i]))) {
        waits.push_back(arguments[i]);
      }
      if (i) {
        names += result ? " and the result " : ", ";
        insts += ", ";
      } else if (result) {
        names += "the result ";
      }
      names += Name(arguments[i]);
      insts += InstText(map[arguments[i]]);
    }
    return Error(goal,
        "mode error in the call to " + what + ": the arguments " + names +
            " have insts " + insts + ", which match none of its modes",
        std::move(waits));
  }
  if (std::optional<Result> error{
          Clobbers(goal, instantiated[chosen], arguments, live, where)}) {
    return *error;
  }
  Pass(instantiated[chosen], arguments, map);
  if (chosen->inferred && !outcomes_.count(chosen)) {
    // The goals after a call of an inferred procedure not checked yet are
    // not reached in this pass, as in mmc, even with final insts known,
    // as for di.
    for (VarId var : arguments) {
      map[var] = insts_.NotReached();
    }
  }
  if (chosen->determinism && NoSolutions(*chosen->determinism)) {
    map.assign(map.size(), insts_.NotReached());
  }
  modes.callee = chosen;
  modes.implied = implied;
  return Ok();
}

// A new procedure for a call of a predicate whose modes are inferred
// (Overton, 3.4.2): its initial insts are the arguments', normalised, and
// its final insts not_reached until its clauses are checked, but for a
// unique argument not used after, which it clobbers, as mmc does, so di.
std::vector<ArgMode> ProcedureChecker::InferredModes(
    llvm::ArrayRef<VarId> arguments, const InstMap &map, const Live &live) {
  std::vector<ArgMode> modes;
  for (VarId var : arguments) {
    const Inst *initial{ops_.Normalise(map[var], TypeOf(var))};
    const Inst *final{insts_.NotReached()};
    if (!live[var] && initial->kind == InstKind::Ground) {
      if (initial->uniqueness == Uniqueness::Unique) {
        final = insts_.Ground(Uniqueness::Clobbered);
      } else if (initial->uniqueness == Uniqueness::MostlyUnique) {
        final = insts_.Ground(Uniqueness::MostlyClobbered);
      }
    }
    modes.push_back({initial, final});
  }
  return modes;
}

const Procedure *ProcedureChecker::Infer(
    const Symbol &callee, llvm::ArrayRef<ArgMode> modes) {
  std::vector<const Procedure *> &list{result_.bySymbol_[&callee]};
  Procedure &procedure{result_.inferred_.emplace_back(Procedure{&callee,
      static_cast<unsigned>(list.size()),
      result_.arena_.SaveArray(modes), std::nullopt, callee.source})};
  procedure.inferred = true;
  list.push_back(&procedure);
  changed_ = true;
  return &procedure;
}

// A call whose inferred mode was rejected waits for other insts, as in
// mmc with --infer-modes; if none come: "in call to predicate `m.p'/1:
// mode error: argument `Y' has the following inst: free which does not
// match any of the valid modes for the callee, because of the following
// error", with the procedure's error after it.
ProcedureChecker::Result ProcedureChecker::RejectedCall(const Goal &goal,
    const Symbol &callee, const Procedure &rejected,
    llvm::ArrayRef<VarId> arguments, const InstMap &map) {
  std::string names, insts;
  std::vector<VarId> waits;
  for (std::size_t i{0}; i < arguments.size(); ++i) {
    VarId var{arguments[i]};
    if (ops_.IsFree(map[var]) || ops_.HasFree(map[var], TypeOf(var))) {
      waits.push_back(var);
    }
    names += (i ? ", " : "") + Name(var);
    insts += (i ? ", " : "") + InstText(map[var]);
  }
  bool one{arguments.size() == 1};
  Result r{Error(goal,
      "mode error in the call to pred " + callee.module.str() + '.' +
          callee.name.str() + '/' + std::to_string(callee.arity) +
          (one ? ": the argument " : ": the arguments ") + names +
          (one ? " has inst " : " have insts ") + insts +
          (one ? ", which matches" : ", which match") +
          " none of its valid modes, because its inferred mode " +
          RejectedText(rejected) + " has a mode error",
      std::move(waits))};
  r.notes.push_back(rejections_.find(&rejected)->second);
  return r;
}

// mmc: "in argument 2 of call to predicate `io.write_int'/3: unique-mode
// error: the called procedure would clobber its argument, but variable
// `IO0' is still live."
std::optional<ProcedureChecker::Result> ProcedureChecker::Clobbers(
    const Goal &goal, llvm::ArrayRef<ArgMode> modes,
    llvm::ArrayRef<VarId> arguments, const Live &live,
    llvm::function_ref<std::string(std::size_t)> where) {
  for (std::size_t i{0}; i < arguments.size() && i < modes.size(); ++i) {
    VarId var{arguments[i]};
    if (live[var] && !LiveAfter(modes[i])) {
      return Error(goal,
          "unique-mode error in " + where(i) +
              ": the called procedure would clobber its argument, but " +
              Name(var) + " is still live",
          {var});
    }
  }
  return std::nullopt;
}

std::vector<ArgMode> ProcedureChecker::Instantiate(llvm::ArrayRef<ArgMode> modes,
    llvm::ArrayRef<VarId> arguments, const InstMap &map) {
  std::vector<const Inst *> actual;
  std::vector<const Type *> types;
  for (VarId var : arguments) {
    actual.push_back(map[var]);
    types.push_back(TypeOf(var));
  }
  return ops_.Instantiate(modes, actual, types, procedure_->parameters);
}

std::optional<std::size_t> ProcedureChecker::Misfit(
    llvm::ArrayRef<ArgMode> modes, llvm::ArrayRef<VarId> arguments,
    const InstMap &map, bool implied) {
  for (std::size_t i{0}; i < arguments.size(); ++i) {
    VarId var{arguments[i]};
    const ArgMode &mode{modes[i]};
    if (ops_.Matches(map[var], mode.initial, TypeOf(var))) {
      continue;
    }
    if (!implied || !ops_.IsFree(mode.initial) || ops_.IsFree(map[var]) ||
        !ops_.Unify(map[var], mode.final, TypeOf(var))) {
      return i;
    }
  }
  return std::nullopt;
}

ProcedureChecker::Result ProcedureChecker::ArgumentError(const Goal &goal,
    llvm::ArrayRef<ArgMode> modes, llvm::ArrayRef<VarId> arguments,
    const InstMap &map, llvm::function_ref<std::string(std::size_t)> where) {
  // mmc: "in argument 1 of call to predicate `m.w'/2: mode error: variable
  // `Y' has instantiatedness `free', expected instantiatedness was
  // `ground'". A bound argument for a free one is not the error, as an
  // implied mode, as in mmc.
  std::optional<std::size_t> implied{Misfit(modes, arguments, map, true)};
  std::size_t i{implied ? *implied : *Misfit(modes, arguments, map, false)};
  VarId var{arguments[i]};
  std::vector<VarId> waits{var};
  if (std::optional<std::size_t> j{Misfit(modes, arguments, map, false)};
      j && arguments[*j] != var) {
    waits.push_back(arguments[*j]);
  }
  return Error(goal,
      "mode error in " + where(i) + ": " + Name(var) + " has inst " +
          InstText(map[var]) + ", but " + Expected(modes[i].initial, map[var]) +
          " is expected",
      std::move(waits));
}

void ProcedureChecker::Pass(llvm::ArrayRef<ArgMode> modes,
    llvm::ArrayRef<VarId> arguments, InstMap &map) {
  for (std::size_t i{0}; i < arguments.size(); ++i) {
    VarId var{arguments[i]};
    const ArgMode &mode{modes[i]};
    if (ops_.IsFree(mode.initial) && ops_.IsFree(map[var])) {
      map[var] = mode.final;
    } else if (const Inst *unified{
                   ops_.Unify(map[var], mode.final, TypeOf(var))}) {
      map[var] = unified;
    } else {
      map[var] = mode.final;
    }
  }
}

// mmc: "In unification of `F' with `G': mode error: attempt at
// higher-order unification. Cannot unify two terms of type `((func int) =
// int)'." (reference manual, "Builtin higher-order insts and modes").
ProcedureChecker::Result ProcedureChecker::HigherOrderUnification(
    const Goal &goal, VarId x, const std::string &other) {
  return Error(goal,
      "mode error in the unification of " + Name(x) + " and " + other +
          ": attempt at higher-order unification: two terms of type " +
          TypeToString(*TypeOf(x)) + " cannot be unified",
      {x});
}

// mmc: "mode error: variable `Z' has instantiatedness `free', expected
// instantiatedness for non-local variables of lambda goals is `ground'."
ProcedureChecker::Result ProcedureChecker::NonlocalError(
    const Goal &goal, const InstMap &map, std::vector<VarId> waits) {
  VarId var{waits.front()};
  return Error(goal,
      "mode error: " + Name(var) + " has inst " + InstText(map[var]) +
          ", but the nonlocal variables of a lambda expression must be ground",
      std::move(waits));
}

bool ProcedureChecker::IsSubtype(const Type *s, const Type *t, unsigned depth) {
  if (s == t) {
    return true;
  }
  if (!s || !t || depth > 64 || s->kind != TypeKind::Named ||
      t->kind != TypeKind::Named) {
    return false;
  }
  if (s->constructor == t->constructor) {
    for (auto [x, y] : llvm::zip(s->arguments, t->arguments)) {
      if (!IsSubtype(x, y, depth + 1)) {
        return false;
      }
    }
    return true;
  }
  const Type *super{types_.SupertypeOf(*s->constructor)};
  if (!super) {
    return false;
  }
  return IsSubtype(types_.types().Substitute(super,
                       [&](unsigned var) {
                         return var < s->arguments.size() ? s->arguments[var]
                                                          : nullptr;
                       }),
      t, depth + 1);
}

// Y = coerce(X) waits for X to be bound, and gives Y the inst of X's
// value in Y's type, or is a mode error (reference manual, "Type
// conversions"). It is an assignment otherwise: X keeps its inst, shared
// if X is used after. With Y bound, as in mmc, the converted value is
// unified with it.
ProcedureChecker::Result ProcedureChecker::CheckCoerce(
    const Unify &unify, const Goal &goal, InstMap &map, const Live &live) {
  VarId x{unify.arguments[0]}, y{unify.left};
  const Type *from{TypeOf(x)}, *to{TypeOf(y)};
  if (ops_.HasFree(map[x], from)) {
    // mmc: "in coerce expression: mode error: the argument term has
    // instantiatedness `free', but it must have a ground inst."
    return Error(goal,
        "mode error in the coerce expression: its argument " + Name(x) +
            " has inst " + InstText(map[x]) + ", but it must be bound",
        {x});
  }
  CoerceFailure failure;
  const Inst *value{ops_.Coerce(map[x], from, to,
      [&](const Type *s, const Type *t) { return IsSubtype(s, t); },
      failure)};
  if (!value) {
    // mmc: "in coerce expression: mode error: in the argument term: in
    // the first argument of function symbol `p': cannot convert the
    // subterm from type `m.fruit' to `m.citrus' because its
    // instantiatedness includes the function symbol `apple'."
    static const char *const ordinals[]{"first", "second", "third", "fourth",
        "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"};
    std::string where;
    for (auto [functor, j] : failure.path) {
      std::string argument{functor.arity == 1 ? "the argument"
              : j < std::size(ordinals)
              ? std::string{"the "} + ordinals[j] + " argument"
              : "argument " + std::to_string(j + 1)};
      where += "in " + argument + " of function symbol " +
          functor.name.str() + ": ";
    }
    std::string why{failure.functor
            ? "its inst includes the function symbol " +
                failure.functor->name.str()
            : "it has inst " + InstText(failure.inst) +
                (ops_.IsGround(failure.inst, failure.from)
                        ? ", and " + TypeToString(*failure.from) +
                            " is not a subtype of " +
                            TypeToString(*failure.to)
                        : "")};
    return Error(goal,
        "mode error in the coerce expression: " + where +
            "cannot convert the " +
            (where.empty() ? "argument " + Name(x) : std::string{"subterm"}) +
            " from type " + TypeToString(*failure.from) + " to " +
            TypeToString(*failure.to) + ", because " + why);
  }
  GoalModes &modes{goals_[unify.number]};
  if (live[x]) {
    map[x] = insts_.Share(map[x]);
    value = insts_.Share(value);
  }
  if (ops_.IsFree(map[y])) {
    map[y] = value;
    modes.unify = UnifyKind::Construct;
  } else {
    const Inst *unified{ops_.Unify(map[y], value, to)};
    if (!unified) {
      return Error(goal,
          "mode error in the unification of " + Name(y) +
              " and the coerce expression: " + Name(y) + " has inst " +
              InstText(map[y]) + ", and the coerce expression has inst " +
              InstText(value),
          {y});
    }
    if (ops_.IsNotReached(unified) && !ops_.IsNotReached(map[y]) &&
        !ops_.IsNotReached(value)) {
      // mmc: "in coerce expression: warning: unification of `Y' and `V_4'
      // cannot succeed."
      warnings_.emplace_back(Source(unify.tokens),
          "the unification of " + Name(y) +
              " and the coerce expression cannot succeed: " + Name(y) +
              " has inst " + InstText(map[y]) +
              ", and the coerce expression has inst " + InstText(value));
    }
    map[y] = unified;
    modes.unify = UnifyKind::Test;
  }
  last_ = Start(unify.tokens);
  return Ok();
}

// A lambda expression's nonlocals must be bound, and ground unless it is an
// any_pred or any_func, and it waits for them to be. Its body is then
// checked as a clause is, from its arguments' initial insts and the
// nonlocals' insts, and its arguments' final insts must match their
// modes, or the lambda expression has a mode error, at
// the goal in it or at the lambda expression, as in mmc. The term gets the
// higher-order inst of the lambda's modes, and the nonlocals are shared
// with it.
ProcedureChecker::Result ProcedureChecker::CheckLambda(
    const Unify &unify, const Goal &goal, InstMap &map, const Live &live) {
  const Lambda &lambda{*unify.functor.lambda};
  VarId x{unify.left};
  if (!ops_.IsFree(map[x])) {
    return HigherOrderUnification(goal, x, "a lambda expression");
  }
  std::optional<std::vector<ArgMode>> modes{
      declarations_.LambdaModes(lambda)};
  if (!modes) {
    return Unsupported(Unchecked::InstParameters);
  }
  // An any_pred or any_func lambda's nonlocals need only be bound, and
  // may be any; a pred or func lambda's may not contain any.
  std::vector<VarId> nonlocals, waits, anys;
  for (VarId var : form_->nonlocals[unify.number]) {
    if (var != x) {
      nonlocals.push_back(var);
      if (ops_.HasFree(map[var], TypeOf(var))) {
        waits.push_back(var);
      } else if (!lambda.any && ops_.HasAny(map[var], TypeOf(var))) {
        anys.push_back(var);
      }
    }
  }
  if (!waits.empty()) {
    return NonlocalError(goal, map, std::move(waits));
  }
  if (!anys.empty()) {
    // mmc: "purity error: lambda is `ground' but contains the following
    // non-local variables whose insts contain `any': X."
    std::string names;
    for (VarId var : anys) {
      names += (names.empty() ? "" : ", ") + Name(var);
    }
    return Error(goal,
        "purity error: the lambda expression is ground, but these nonlocals "
        "have insts that contain any: " + names);
  }
  InstMap inner{map};
  Live inside(live.size(), false);
  for (VarId var : nonlocals) {
    inside[var] = true;
  }
  for (std::size_t i{0}; i < lambda.arguments.size(); ++i) {
    inner[lambda.arguments[i]] = (*modes)[i].initial;
    inside[lambda.arguments[i]] = LiveAfter((*modes)[i]);
  }
  Result r{Check(lambda.body, inner, inside)};
  if (r.outcome == Outcome::Ok) {
    for (std::size_t i{0}; i < lambda.arguments.size(); ++i) {
      VarId var{lambda.arguments[i]};
      if (!ops_.Matches(inner[var], (*modes)[i].final, TypeOf(var))) {
        // mmc names the argument LambdaHeadVar__1.
        r = Error(goal,
            FinalError("argument " + std::to_string(i + 1) +
                    ", LambdaHeadVar__" + std::to_string(i + 1) + ",",
                inner[var], (*modes)[i].final, TypeOf(var)));
        break;
      }
    }
  }
  if (r.outcome == Outcome::Error) {
    // Nothing can make it run, as in mmc ("waiting on { }").
    r.waits.clear();
    r.never = true;
    r.lambda = true;
  }
  if (r.outcome != Outcome::Ok) {
    return r;
  }
  for (VarId var : nonlocals) {
    map[var] = insts_.Share(map[var]);
  }
  map[x] = insts_.HigherOrder(lambda.kind, lambda.any, *modes,
      lambda.determinism.value_or(parser::Determinism::Det),
      Uniqueness::Unique);
  GoalModes &g{goals_[unify.number]};
  g.unify = UnifyKind::Construct;
  g.determinism = lambda.determinism.value_or(parser::Determinism::Det);
  g.lambda = result_.arena_.SaveArray(llvm::ArrayRef<ArgMode>{*modes});
  last_ = Start(unify.tokens);
  return Ok();
}

// A higher-order term made from a predicate's or function's name and its
// first arguments, of one of its procedures (reference manual, "Creating
// higher-order terms"): of a predicate or function with several, the one
// that the insts of the arguments of higher-order types match, as in mmc.
// The arguments are then passed as a call's are, must be ground, and are
// shared with the term.
ProcedureChecker::Result ProcedureChecker::CheckClosure(
    const Unify &unify, const Goal &goal, const Symbol &symbol, InstMap &map) {
  VarId x{unify.left};
  llvm::ArrayRef<VarId> arguments{unify.arguments};
  if (!ops_.IsFree(map[x])) {
    return HigherOrderUnification(goal, x, "a higher-order term");
  }
  llvm::ArrayRef<const Procedure *> procedures{ProceduresFor(symbol)};
  if (procedures.empty()) {
    return Unsupported(Unchecked::InferredModes);
  }
  bool function{symbol.kind == SymbolKind::Function};
  std::string what{std::string{function ? "func " : "pred "} +
      symbol.module.str() + '.' + symbol.name.str() + '/' +
      std::to_string(symbol.arity)};
  const Procedure *chosen{procedures.front()};
  if (procedures.size() > 1) {
    std::vector<VarId> higher;
    for (VarId var : arguments) {
      if (TypeOf(var) && TypeOf(var)->kind == TypeKind::HigherOrder) {
        higher.push_back(var);
      }
    }
    std::vector<const Procedure *> candidates;
    for (const Procedure *p : procedures) {
      bool fits{true};
      for (std::size_t i{0}; i < arguments.size(); ++i) {
        VarId var{arguments[i]};
        if (llvm::is_contained(higher, var) &&
            !ops_.Matches(map[var], p->modes[i].initial, TypeOf(var))) {
          fits = false;
        }
      }
      if (fits) {
        candidates.push_back(p);
      }
    }
    if (candidates.size() != 1) {
      // mmc: "mode error in unification of `P' and higher-order term
      // based on multi-moded predicate `m.app'/2. The modes of the
      // argument variables match more than one of the called predicate's
      // modes." With none, "The mode of the argument variable does not
      // match any of the called predicate's modes", or "The modes of the
      // argument variables do not".
      return Error(goal,
          "mode error in the unification of " + Name(x) +
              " and a higher-order term based on the multi-moded " + what +
              ": " +
              (!candidates.empty() ? "the modes of the argument variables "
                                     "match more than one of its modes"
                      : arguments.size() == 1
                      ? "the mode of the argument variable does not match "
                        "any of its modes"
                      : "the modes of the argument variables do not match "
                        "any of its modes"),
          std::move(higher));
    }
    chosen = candidates.front();
  }
  if (!chosen->determinism) {
    // mmc: "Error: the predicate `m.p' has no declared determinism, so a
    // curried call to it may not be used as a lambda expression."
    return Error(goal,
        "the " + what +
            " has no declared determinism, so a curried call to it may not "
            "be used as a lambda expression");
  }
  // mmc makes the term a lambda expression that calls the procedure, with
  // the remaining arguments in the procedure's own modes, so they are
  // passed from their initial insts, and must get their final insts,
  // with the inst parameters that the curried arguments bound.
  std::vector<const Inst *> actual;
  std::vector<const Type *> types;
  for (std::size_t i{0}; i < chosen->modes.size(); ++i) {
    actual.push_back(
        i < arguments.size() ? map[arguments[i]] : chosen->modes[i].initial);
    types.push_back(i < arguments.size() ? TypeOf(arguments[i]) : nullptr);
  }
  std::vector<ArgMode> modes{ops_.Instantiate(chosen->modes, actual, types)};
  for (std::size_t i{arguments.size()}; i < modes.size(); ++i) {
    const ArgMode &own{chosen->modes[i]};
    const Inst *final{ops_.IsFree(own.initial)
            ? modes[i].final
            : ops_.Unify(own.initial, modes[i].final, nullptr)};
    std::string subject{"argument " + std::to_string(i - arguments.size() + 1) +
        " of the higher-order term"};
    Result r{Ok()};
    if (!ops_.Matches(own.initial, modes[i].initial, nullptr)) {
      // mmc reports this at the call in the lambda expression.
      r = Error(goal,
          "mode error in argument " + std::to_string(i + 1) +
              " of the call to " + what + ": " + subject + " has inst " +
              InstText(own.initial) + ", but " +
              Expected(modes[i].initial, own.initial) + " is expected");
    } else if (!final || !ops_.Matches(final, own.final, nullptr)) {
      r = Error(goal,
          FinalError(subject, final ? final : modes[i].final, own.final,
              nullptr));
    }
    if (r.outcome == Outcome::Error) {
      r.waits.clear();
      r.never = true;
      return r;
    }
  }
  bool implied{false};
  if (Misfit(modes, arguments, map, false)) {
    if (Misfit(modes, arguments, map, true)) {
      return ArgumentError(goal, modes, arguments, map, [&](std::size_t i) {
        return "argument " + std::to_string(i + 1) + " of the call to " + what;
      });
    }
    implied = true;
  }
  // The arguments are then the nonlocals of a lambda expression, and must
  // be ground, as in mmc.
  std::vector<VarId> waits;
  for (VarId var : arguments) {
    if (!ops_.IsGround(map[var], TypeOf(var))) {
      waits.push_back(var);
    }
  }
  if (!waits.empty()) {
    return NonlocalError(goal, map, std::move(waits));
  }
  Pass(modes, arguments, map);
  for (VarId var : arguments) {
    map[var] = insts_.Share(map[var]);
  }
  map[x] = insts_.HigherOrder(
      function ? parser::PredOrFunc::Function : parser::PredOrFunc::Predicate,
      false, chosen->modes.drop_front(arguments.size()), *chosen->determinism,
      Uniqueness::Unique);
  GoalModes &g{goals_[unify.number]};
  g.unify = UnifyKind::Construct;
  g.callee = chosen;
  g.implied = implied;
  last_ = Start(unify.tokens);
  return Ok();
}

// A call of a higher-order term, in the modes of its higher-order inst, or
// for a function without one, of the default inst (reference manual,
// "Calling higher-order terms" and "Default insts for functions"). Its
// arguments are passed as a call's are.
ProcedureChecker::Result ProcedureChecker::CheckHigherOrderCall(
    const Goal &goal, const HOCall &call, InstMap &map, const Live &live) {
  bool function{call.kind == parser::PredOrFunc::Function};
  std::size_t arity{call.arguments.size() - (function ? 1 : 0)};
  const char *kind{function ? "function" : "predicate"};
  const Inst *inst{ops_.Plain(map[call.closure])};
  if (function && inst->kind == InstKind::Ground) {
    inst = ops_.DefaultFunction(arity);
  }
  if (inst->kind != InstKind::HigherOrder || inst->predOrFunc != call.kind ||
      inst->modes.size() != call.arguments.size()) {
    // mmc: "in argument 1 (i.e. the predicate term) of higher-order
    // predicate call: mode error: variable `P' has instantiatedness
    // `ground', expecting higher-order pred inst of arity 1."
    return Error(goal,
        std::string{"mode error in argument 1 (the "} + kind +
            " term) of the higher-order " + kind +
            " call: " + Name(call.closure) + " has inst " +
            InstText(map[call.closure]) + ", but a higher-order " +
            (function ? "func" : "pred") + " inst of arity " +
            std::to_string(arity) + " is expected",
        {call.closure});
  }
  // mmc takes an inst variable of the term's inst to be the procedure's
  // own of the same number, if it has one, whatever its name, so that it
  // is not bound by the arguments (see the design document's register).
  std::vector<ArgMode> own;
  auto identify{[&](const Inst &parameter) {
    for (const Inst *mine : procedure_->parameters) {
      if (mine->number == parameter.number) {
        return mine;
      }
    }
    return &parameter;
  }};
  for (const ArgMode &mode : inst->modes) {
    own.push_back({insts_.Substitute(mode.initial, identify),
        insts_.Substitute(mode.final, identify)});
  }
  std::vector<ArgMode> modes{Instantiate(own, call.arguments, map)};
  bool implied{false};
  if (Misfit(modes, call.arguments, map, false)) {
    if (Misfit(modes, call.arguments, map, true)) {
      // mmc: "in argument 2 (i.e. argument 1 of the called predicate) of
      // higher-order predicate call".
      return ArgumentError(
          goal, modes, call.arguments, map, [&](std::size_t i) {
            bool result{function && i + 1 == call.arguments.size()};
            return "argument " + std::to_string(i + 2) + " (" +
                (result ? std::string{"the result of the called function"}
                        : "argument " + std::to_string(i + 1) +
                            " of the called " + kind) +
                ") of the higher-order " + kind + " call";
          });
    }
    implied = true;
  }
  if (std::optional<Result> error{
          Clobbers(goal, modes, call.arguments, live, [&](std::size_t i) {
            return "argument " + std::to_string(i + 2) +
                " of the higher-order " + kind + " call";
          })}) {
    return *error;
  }
  Pass(modes, call.arguments, map);
  if (NoSolutions(inst->determinism)) {
    map.assign(map.size(), insts_.NotReached());
  }
  GoalModes &g{goals_[call.number]};
  g.unify = UnifyKind::Call;
  g.implied = implied;
  g.determinism = inst->determinism;
  g.lambda = result_.arena_.SaveArray(llvm::ArrayRef<ArgMode>{modes});
  last_ = Start(call.tokens);
  return Ok();
}


void DumpClauseModes(llvm::raw_ostream &os, const ClauseModes &modes,
    const parser::TokenSequence &tokens,
    llvm::function_ref<std::string(parser::CharBlock)> position) {
  const ClauseForm &form{*modes.clause->clause};
  auto at{[&](parser::TokenRange range) {
    return position(tokens.tokens[range.first].spelling);
  }};
  std::function<void(const Goal &, unsigned)> dump{[&](const Goal &goal,
                                                      unsigned depth) {
    const GoalModes &g{modes.goals[NumberOf(goal)]};
    std::string indent(2 * depth, ' ');
    auto part{[&](const char *name, const Goal &inner) {
      os << indent << "      " << name << '\n';
      dump(inner, depth + 2);
    }};
    common::visit(
        common::visitors{
            [&](const Conj *) {
              for (const Goal &inner : g.order) {
                dump(inner, depth);
              }
            },
            [&](const Disj *disj) {
              os << indent << "    " << at(disj->tokens)
                 << (disj->goals.empty() ? " fail\n" : " disjunction\n");
              for (const Goal &branch : disj->goals) {
                part("branch", branch);
              }
            },
            [&](const IfThenElse *ite) {
              os << indent << "    " << at(ite->tokens) << " if-then-else\n";
              part("if", ite->condition);
              part("then", ite->then);
              part("else", ite->otherwise);
            },
            [&](const Not *negation) {
              os << indent << "    " << at(negation->tokens) << " not\n";
              dump(negation->goal, depth + 1);
            },
            [&](const Scope *scope) { dump(scope->goal, depth); },
            [&](const Unify *unify) {
              os << indent << "    " << at(unify->tokens) << ' ';
              switch (g.unify) {
              case UnifyKind::Assign:
                os << "assign " << VarName(form, g.assigned) << " := "
                   << VarName(form,
                          g.assigned == unify->left ? unify->var : unify->left);
                break;
              case UnifyKind::Call:
                os << "call " << *g.callee->symbol << " mode "
                   << g.callee->number << (g.implied ? " implied" : "") << ": "
                   << VarName(form, unify->left) << " = "
                   << TermText(form, *unify);
                break;
              default:
                os << llvm::StringRef{EnumToString(g.unify)}.lower() << ' '
                   << VarName(form, unify->left) << " = "
                   << TermText(form, *unify);
                if (g.callee) {
                  os << ", closure of " << *g.callee->symbol << " mode "
                     << g.callee->number << (g.implied ? " implied" : "");
                }
              }
              os << '\n';
              if (const Lambda * lambda{unify->functor.lambda}) {
                dump(lambda->body, depth + 1);
              }
            },
            [&](const HOCall *call) {
              os << indent << "    " << at(call->tokens) << " higher-order call"
                 << (g.implied ? " implied" : "") << ": ";
              llvm::ArrayRef<VarId> arguments{call->arguments};
              if (call->kind == parser::PredOrFunc::Function) {
                os << VarName(form, arguments.back()) << " = ";
                arguments = arguments.drop_back();
              }
              os << VarName(form, call->closure) << '(';
              llvm::interleave(
                  arguments, os, [&](VarId var) { os << VarName(form, var); },
                  ", ");
              os << ")\n";
            },
            [&](const Call *call) {
              os << indent << "    " << at(call->tokens) << " call "
                 << *g.callee->symbol
                 << " mode " << g.callee->number
                 << (g.implied ? " implied" : "") << ": " << call->name->name
                 << '(';
              llvm::interleave(
                  call->arguments, os,
                  [&](VarId var) { os << VarName(form, var); }, ", ");
              os << ")\n";
            },
            [&](const auto *) { os << indent << "    (not checked)\n"; },
        },
        goal);
  }};
  dump(form.body, 0);
  os << "    final:";
  for (std::size_t i{0}; i < form.head.size() && i < modes.final.size(); ++i) {
    os << ' ' << VarName(form, form.head[i]) << ' '
       << InstToString(*modes.final[i]) << (i + 1 < form.head.size() ? "," : "");
  }
  os << '\n';
}

std::unique_ptr<ModeChecking> CheckModes(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, const SemanticForm &,
    const NameResolution &resolution, TypeChecking &types,
    parser::Messages &messages, const ModeCheckOptions &options) {
  auto result{std::make_unique<ModeChecking>()};
  ModeChecker declarations{
      *result, resolution, types, tokens, messages, options};
  declarations.Run(tree);
  ProcedureChecker{*result, declarations, resolution, types, result->ops(),
      tokens, messages, options}
      .Run();
  return result;
}

} // namespace merkurius::semantics
