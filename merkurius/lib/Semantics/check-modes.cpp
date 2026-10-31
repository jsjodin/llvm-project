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
#include <functional>
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
  if (number >= scope.made.size()) {
    // Unconstrained means compatible with ground.
    const Inst *constraint{scope.constraints[number]};
    scope.made.push_back(insts_.Parameter(
        number, name, constraint ? constraint : insts_.Ground()));
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
      continue; // its modes are inferred from its calls
    }
    std::vector<Procedure> procedures;
    std::vector<parser::CharBlock> sources;
    auto add{[&](Declared declared, parser::CharBlock where) {
      for (std::size_t i{0}; i < procedures.size(); ++i) {
        if (procedures[i].modes == llvm::ArrayRef<ArgMode>{declared.modes}) {
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
        // The default mode: every argument in, the result out, det.
        std::vector<ArgMode> modes(symbol->arity,
            ArgMode{insts_.Ground(), insts_.Ground()});
        modes.push_back(ArgMode{insts_.Free(), insts_.Ground()});
        procedures.push_back(Procedure{symbol, 0,
            result_.arena_.SaveArray(llvm::ArrayRef<ArgMode>{modes}),
            parser::Determinism::Det, {}});
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
  ProcedureChecker(ModeChecking &result, const NameResolution &resolution,
      TypeChecking &types, InstOps &ops, const parser::TokenSequence &tokens,
      parser::Messages &messages)
      : result_{result}, insts_{result.insts_}, resolution_{resolution},
        types_{types}, ops_{ops}, tokens_{tokens}, messages_{messages} {}

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
  };
  static Result Ok() { return {}; }
  static Result Unsupported(Unchecked why) {
    return {Outcome::Unsupported, {}, {}, why};
  }
  Result Error(
      const Goal &goal, std::string text, std::vector<VarId> waits = {}) {
    return {Outcome::Error, Source(TokensOf(goal)),
        "in the clause for " + ProcedureToString(*procedure_) + ": " + text,
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
  Result Check(const Goal &, InstMap &, const Live &);
  Result Conjunction(const Goal &conj, llvm::ArrayRef<Goal>, InstMap &,
      const Live &);
  Result CheckUnify(const Unify &, const Goal &, InstMap &, const Live &);
  Result CheckVarUnify(const Unify &, const Goal &, InstMap &, const Live &);
  Result CheckCall(const Goal &, const Symbol &, llvm::ArrayRef<VarId>,
      bool function, InstMap &, GoalModes &);
  llvm::ArrayRef<const Procedure *> ProceduresFor(const Symbol &);
  Result CheckFinal(const InstMap &, ClauseModes &);
  static void Flatten(const Goal &, std::vector<Goal> &);

  ModeChecking &result_;
  InstContext &insts_;
  const NameResolution &resolution_;
  TypeChecking &types_;
  InstOps &ops_;
  const parser::TokenSequence &tokens_;
  parser::Messages &messages_;

  // The clause being checked, and for which procedure.
  const ClauseTypes *clause_{nullptr};
  const ClauseForm *form_{nullptr};
  const Procedure *procedure_{nullptr};
  std::vector<GoalModes> goals_;
  // The goal run last, where an error in the final insts is reported, as
  // in mmc.
  parser::CharBlock last_{};
  // By variable, how often a construction put it in a term while free.
  std::vector<unsigned> touched_;
  // Warnings of the goals run, reported once the clause is checked.
  std::vector<std::pair<parser::CharBlock, std::string>> warnings_;
  // The procedures of fields, made when first called.
  llvm::DenseMap<const Symbol *, const Procedure *> fields_;
};

void ProcedureChecker::Run() {
  // The clauses of each predicate and function, but not those of
  // instance methods.
  llvm::DenseMap<const Symbol *, std::vector<const ClauseTypes *>> clauses;
  for (const ClauseTypes &clause : types_.clauses()) {
    if (clause.definition && !clause.method) {
      clauses[clause.definition].push_back(&clause);
    }
  }
  for (const Procedure &procedure : result_.procedures_) {
    auto found{clauses.find(procedure.symbol)};
    llvm::ArrayRef<const ClauseTypes *> own;
    if (found != clauses.end()) {
      own = found->second;
    }
    auto unchecked{[&](Unchecked why) {
      result_.unchecked_[&procedure] = why;
      result_.clauses_.erase(&procedure);
    }};
    // mmc checks modes only after the types are right.
    if (llvm::any_of(own, [](const ClauseTypes *c) { return c->typeError; })) {
      unchecked(Unchecked::TypeErrors);
      continue;
    }
    if (llvm::any_of(
            own, [](const ClauseTypes *c) { return c->clause->modeSpecific; })) {
      unchecked(Unchecked::ModeSpecificClauses);
      continue;
    }
    if (!procedure.parameters.empty()) {
      unchecked(Unchecked::InstParameters);
      continue;
    }
    std::vector<ClauseModes> &checked{result_.clauses_[&procedure]};
    for (const ClauseTypes *clause : own) {
      ClauseModes modes{clause, &procedure, {}, {}};
      warnings_.clear();
      Result r{CheckClause(*clause, procedure, modes)};
      if (r.outcome != Outcome::Unsupported) {
        for (const auto &[where, text] : warnings_) {
          messages_.Say(where, "%s"_warn_en_US, text);
        }
      }
      if (r.outcome == Outcome::Unsupported) {
        unchecked(r.why);
        break;
      }
      if (r.outcome == Outcome::Error) {
        // Only the first mode error of a procedure, as in mmc.
        messages_.Say(r.where, "%s"_err_en_US, r.text);
        result_.errors_.insert(&procedure);
        result_.clauses_.erase(&procedure);
        break;
      }
      checked.push_back(modes);
    }
  }
}

ProcedureChecker::Result ProcedureChecker::CheckClause(
    const ClauseTypes &clause, const Procedure &procedure, ClauseModes &out) {
  clause_ = &clause;
  form_ = clause.clause;
  procedure_ = &procedure;
  goals_.assign(form_->goals, GoalModes{});
  touched_.assign(form_->vars.size(), 0);
  last_ = Start(form_->tokens);
  InstMap map(form_->vars.size(), insts_.Free());
  Live live(form_->vars.size(), false);
  for (std::size_t i{0}; i < form_->head.size() && i < procedure.modes.size();
       ++i) {
    map[form_->head[i]] = procedure.modes[i].initial;
    // A head variable is used after the clause, unless its final inst is
    // clobbered, as for di (Overton, 3.2.3).
    const Inst *final{procedure.modes[i].final};
    while (final->kind == InstKind::Defined) {
      final = insts_.Expand(final);
    }
    live[form_->head[i]] = !(
        (final->kind == InstKind::Ground || final->kind == InstKind::Any) &&
        (final->uniqueness == Uniqueness::Clobbered ||
            final->uniqueness == Uniqueness::MostlyClobbered));
  }
  Result r{Check(form_->body, map, live)};
  if (r.outcome != Outcome::Ok) {
    return r;
  }
  r = CheckFinal(map, out);
  if (r.outcome != Outcome::Ok) {
    return r;
  }
  out.goals = result_.arena_.SaveArray(llvm::ArrayRef<GoalModes>{goals_});
  return Ok();
}

// The head's insts at the end must match the final insts of the
// procedure's modes.
ProcedureChecker::Result ProcedureChecker::CheckFinal(
    const InstMap &map, ClauseModes &out) {
  std::vector<const Inst *> final;
  for (std::size_t i{0}; i < form_->head.size(); ++i) {
    VarId var{form_->head[i]};
    final.push_back(map[var]);
    if (i >= procedure_->modes.size()) {
      continue;
    }
    const Inst *expected{procedure_->modes[i].final};
    const Type *type{TypeOf(var)};
    if (ops_.Matches(map[var], expected, type)) {
      continue;
    }
    // mmc: "argument 1 did not get sufficiently instantiated", "became
    // too instantiated", or "had the wrong instantiatedness", with
    // "Final instantiatedness of `X' was `free', expected final
    // instantiatedness was `ground'".
    const char *how{ops_.HasFree(map[var], type) ? "did not get sufficiently instantiated"
            : ops_.HasFree(expected, type)      ? "became too instantiated"
                                                : "had the wrong inst"};
    return {Outcome::Error, last_,
        "in the clause for " + ProcedureToString(*procedure_) +
            ": mode error: argument " + std::to_string(i + 1) + ", " +
            Name(var) + ", " + how + ": its final inst is " +
            InstText(map[var]) + ", but " + InstText(expected) +
            " is expected"};
  }
  out.final = result_.arena_.SaveArray(llvm::ArrayRef<const Inst *>{final});
  return Ok();
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

ProcedureChecker::Result ProcedureChecker::Check(
    const Goal &goal, InstMap &map, const Live &live) {
  return common::visit(
      common::visitors{
          [&](const Conj *conj) -> Result {
            if (conj->parallel) {
              return Unsupported(Unchecked::ParallelConjunction);
            }
            std::vector<Goal> goals;
            Flatten(goal, goals);
            return Conjunction(goal, goals, map, live);
          },
          [&](const Disj *) { return Unsupported(Unchecked::Disjunction); },
          [&](const IfThenElse *) {
            return Unsupported(Unchecked::IfThenElse);
          },
          [&](const Not *) { return Unsupported(Unchecked::Negation); },
          [&](const Unify *unify) { return CheckUnify(*unify, goal, map, live); },
          [&](const Call *call) -> Result {
            const Symbol *callee{clause_->symbols[call->number]};
            if (!callee || callee->kind != SymbolKind::Predicate ||
                callee->arity != call->arguments.size()) {
              return Unsupported(Unchecked::HigherOrder);
            }
            Result r{CheckCall(goal, *callee, call->arguments, false, map,
                goals_[call->number])};
            if (r.outcome == Outcome::Ok) {
              goals_[call->number].unify = UnifyKind::Call;
              last_ = Start(call->tokens);
            }
            return r;
          },
          [&](const HOCall *) { return Unsupported(Unchecked::HigherOrder); },
          [&](const Scope *scope) -> Result {
            switch (scope->kind) {
            case ScopeKind::Trace:
            case ScopeKind::PromiseSolutions:
              return Unsupported(Unchecked::Scope);
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
  // The insts of the variables a goal waits for, and how often each was
  // put in a term free; mmc wakes a goal only when these change.
  auto snapshot{[&](const Goal &goal, const Result &error) {
    std::vector<const Inst *> insts;
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
    Result r{Check(goal, trial, after)};
    if (r.outcome != Outcome::Ok) {
      last_ = last;
      warnings_.resize(warnings);
      touched_ = std::move(touched);
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
  while (true) {
    Goal goal;
    if (!agenda.empty()) {
      goal = agenda.front();
      agenda.pop_front();
    } else if (next < goals.size()) {
      goal = goals[next++];
    } else {
      break;
    }
    Result r{run(goal)};
    if (r.outcome == Outcome::Unsupported) {
      return r;
    }
    if (r.outcome == Outcome::Error) {
      std::vector<const Inst *> insts{snapshot(goal, r)};
      waiting.push_back({goal, std::move(insts), std::move(r)});
      continue;
    }
    wake();
  }
  if (!waiting.empty()) {
    return waiting.front().error;
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
    return Ok();
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
    warnings_.emplace_back(Source(unify.tokens),
        "the unification of " + Name(x) + " and " + Name(y) +
            " cannot succeed: " + Name(x) + " has inst " + InstText(a) +
            ", and " + Name(y) + " has inst " + InstText(b));
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
    return Unsupported(Unchecked::Lambda);
  case FunctorKind::Coerce:
    return Unsupported(Unchecked::Coerce);
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
    if (symbol->kind == SymbolKind::Function ||
        symbol->kind == SymbolKind::Field ||
        symbol->kind == SymbolKind::FieldUpdate) {
      if (symbol->arity != unify.arguments.size()) {
        return Unsupported(Unchecked::HigherOrder);
      }
      std::vector<VarId> arguments{
          unify.arguments.begin(), unify.arguments.end()};
      arguments.push_back(x);
      r = CheckCall(goal, *symbol, arguments, true, map, modes);
      if (r.outcome == Outcome::Ok) {
        modes.unify = UnifyKind::Call;
        last_ = Start(unify.tokens);
      }
      return r;
    }
    if (symbol->kind != SymbolKind::Constructor ||
        symbol->arity != unify.arguments.size()) {
      return Unsupported(Unchecked::HigherOrder);
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
      map[unify.arguments[i]] = shared[i];
      // An argument with free parts is now part of the term, which wakes
      // the goals that wait for it, as in mmc.
      if (ops_.HasFree(shared[i], TypeOf(unify.arguments[i]))) {
        ++touched_[unify.arguments[i]];
      }
    }
    map[x] = term;
    modes.unify = UnifyKind::Construct;
    last_ = Start(unify.tokens);
    return Ok();
  }
  // A deconstruction, or a test if the arguments are bound too.
  const Inst *term{insts_.Bound(Uniqueness::Shared, {{functor, arguments}})};
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
    warnings_.emplace_back(Source(unify.tokens),
        "the unification of " + Name(x) + " and " + TermText(*form_, unify) +
            " cannot succeed: " + Name(x) + " has inst " + InstText(a));
  }
  bool test{true};
  for (std::size_t i{0}; i < unify.arguments.size(); ++i) {
    test &= !ops_.IsFree(arguments[i]);
    map[unify.arguments[i]] = parts ? (*parts)[i] : insts_.NotReached();
  }
  map[x] = unified;
  modes.unify = test ? UnifyKind::Test : UnifyKind::Deconstruct;
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
    InstMap &map, GoalModes &modes) {
  llvm::ArrayRef<const Procedure *> procedures{ProceduresFor(callee)};
  if (procedures.empty()) {
    // Its modes are to be inferred, or its missing mode declaration was
    // reported.
    return Unsupported(Unchecked::InferredModes);
  }
  if (llvm::any_of(procedures,
          [](const Procedure *p) { return !p->parameters.empty(); })) {
    return Unsupported(Unchecked::InstParameters);
  }
  // With `implied`, a bound argument fits an output too: the call outputs
  // to a fresh variable, which is then unified with it (an implied mode,
  // Overton 3.3.2). The argument may have free parts, of variables not
  // used again, which the unification binds, as in mmc. The first
  // argument that does not fit, if any.
  auto misfit{[&](const Procedure &procedure,
                  bool implied) -> std::optional<std::size_t> {
    for (std::size_t i{0}; i < arguments.size(); ++i) {
      VarId var{arguments[i]};
      const ArgMode &mode{procedure.modes[i]};
      if (ops_.Matches(map[var], mode.initial, TypeOf(var))) {
        continue;
      }
      if (!implied || !ops_.IsFree(mode.initial) || ops_.IsFree(map[var]) ||
          !ops_.Unify(map[var], mode.final, TypeOf(var))) {
        return i;
      }
    }
    return std::nullopt;
  }};
  auto moreSpecific{[&](const Procedure &p, const Procedure &q) {
    for (std::size_t i{0}; i < arguments.size(); ++i) {
      if (!ops_.Matches(p.modes[i].initial, q.modes[i].initial,
              TypeOf(arguments[i]))) {
        return false;
      }
    }
    return true;
  }};
  std::vector<const Procedure *> candidates;
  bool implied{false};
  for (bool allowImplied : {false, true}) {
    for (const Procedure *procedure : procedures) {
      if (procedure->modes.size() == arguments.size() &&
          !misfit(*procedure, allowImplied)) {
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
  std::string what{std::string{function ? "func " : "pred "} +
      callee.module.str() + '.' + callee.name.str() + '/' +
      std::to_string(callee.arity)};
  if (!chosen) {
    if (procedures.size() == 1 &&
        procedures.front()->modes.size() == arguments.size()) {
      // mmc: "in argument 1 of call to predicate `m.w'/2: mode error:
      // variable `Y' has instantiatedness `free', expected instantiatedness
      // was `ground'".
      const Procedure &only{*procedures.front()};
      // The call waits for the first argument that does not fit, with or
      // without an implied mode, as in mmc.
      std::vector<VarId> waits{arguments[*misfit(only, false)]};
      if (VarId var{arguments[*misfit(only, true)]}; var != waits.front()) {
        waits.push_back(var);
      }
      for (std::size_t i{0}; i < arguments.size(); ++i) {
        VarId var{arguments[i]};
        if (!ops_.Matches(map[var], only.modes[i].initial, TypeOf(var))) {
          bool result{function && i + 1 == arguments.size()};
          return Error(goal,
              "mode error in " +
                  (result ? std::string{"the result"}
                          : "argument " + std::to_string(i + 1)) +
                  " of the call to " + what + ": " + Name(var) +
                  " has inst " + InstText(map[var]) + ", but " +
                  InstText(only.modes[i].initial) + " is expected",
              std::move(waits));
        }
      }
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
  for (std::size_t i{0}; i < arguments.size(); ++i) {
    VarId var{arguments[i]};
    const ArgMode &mode{chosen->modes[i]};
    if (ops_.IsFree(mode.initial) && ops_.IsFree(map[var])) {
      map[var] = mode.final;
    } else if (const Inst *unified{
                   ops_.Unify(map[var], mode.final, TypeOf(var))}) {
      map[var] = unified;
    } else {
      map[var] = mode.final;
    }
  }
  modes.callee = chosen;
  modes.implied = implied;
  return Ok();
}

void DumpClauseModes(llvm::raw_ostream &os, const ClauseModes &modes,
    const parser::TokenSequence &tokens,
    llvm::function_ref<std::string(parser::CharBlock)> position) {
  const ClauseForm &form{*modes.clause->clause};
  auto at{[&](parser::TokenRange range) {
    return position(tokens.tokens[range.first].spelling);
  }};
  std::function<void(const Goal &)> dump{[&](const Goal &goal) {
    const GoalModes &g{modes.goals[NumberOf(goal)]};
    common::visit(
        common::visitors{
            [&](const Conj *) {
              for (const Goal &inner : g.order) {
                dump(inner);
              }
            },
            [&](const Scope *scope) { dump(scope->goal); },
            [&](const Unify *unify) {
              os << "    " << at(unify->tokens) << ' ';
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
              }
              os << '\n';
            },
            [&](const Call *call) {
              os << "    " << at(call->tokens) << " call " << *g.callee->symbol
                 << " mode " << g.callee->number
                 << (g.implied ? " implied" : "") << ": " << call->name->name
                 << '(';
              llvm::interleave(
                  call->arguments, os,
                  [&](VarId var) { os << VarName(form, var); }, ", ");
              os << ")\n";
            },
            [&](const auto *) { os << "    (not checked)\n"; },
        },
        goal);
  }};
  dump(form.body);
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
  ModeChecker{*result, resolution, types, tokens, messages, options}.Run(tree);
  ConstructorTable constructors{resolution, types};
  InstOps ops{result->insts(), constructors};
  ProcedureChecker{*result, resolution, types, ops, tokens, messages}.Run();
  return result;
}

} // namespace merkurius::semantics
