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

llvm::ArrayRef<Procedure> ModeChecking::ProceduresOf(const Symbol &symbol) const {
  auto it{bySymbol_.find(&symbol)};
  if (it == bySymbol_.end()) {
    return {};
  }
  return llvm::ArrayRef<Procedure>{procedures_}.slice(
      it->second.first, it->second.second);
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
    return {tokens_.tokens[range.first].spelling.begin(),
        tokens_.tokens[range.last].spelling.end()};
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
  const Symbol *ModeDeclTarget(
      const parser::ModeDecl &, const Symbol *typeclass);
  void CheckForTypes(
      const Symbol &, llvm::ArrayRef<ArgMode>, parser::CharBlock where);

  ModeChecking &result_;
  InstContext &insts_;
  const NameResolution &resolution_;
  TypeChecking &types_;
  const parser::TokenSequence &tokens_;
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

const Symbol *ModeChecker::ModeDeclTarget(
    const parser::ModeDecl &decl, const Symbol *typeclass) {
  parser::PredOrFunc kind{decl.result ? parser::PredOrFunc::Function
                                      : parser::PredOrFunc::Predicate};
  SymbolKind symbolKind{decl.result ? SymbolKind::Function
                                    : SymbolKind::Predicate};
  unsigned arity{static_cast<unsigned>(decl.arguments.size())};
  const ModuleSymbols &own{resolution_.module()};
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

  // The mode declarations of each predicate and function, with the
  // section each is in.
  struct ModeItem {
    const parser::ModeDecl *decl;
    bool interface;
  };
  llvm::DenseMap<const Symbol *, std::vector<ModeItem>> modeDecls;
  bool interface{true};
  auto addMode{[&](const parser::ModeDecl &decl, const Symbol *typeclass) {
    if (const Symbol *target{ModeDeclTarget(decl, typeclass)}) {
      modeDecls[target].push_back({&decl, interface});
    }
  }};
  for (const parser::Item &item : tree.items) {
    if (const auto *section{std::get_if<const parser::SectionDecl *>(&item)}) {
      interface = (*section)->section == parser::Section::Interface;
    } else if (const auto *decl{std::get_if<const parser::ModeDecl *>(&item)}) {
      addMode(**decl, nullptr);
    } else if (const auto *typeclass{
                   std::get_if<const parser::TypeclassDecl *>(&item)}) {
      const Symbol *symbol{nullptr};
      for (const Symbol *s : own.Lookup((*typeclass)->name->name)) {
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

  for (const Symbol *symbol : own.symbols()) {
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
          messages_
              .Say(sources[i], "%s has the same mode declared twice"_err_en_US,
                  Text(*symbol))
              .Attach(where, "the other declaration of the mode is here"_because_en_US);
          return;
        }
      }
      CheckForTypes(*symbol, declared.modes, where);
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
      if (symbol->exported && !declared.determinism) {
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
      if (combined) {
        // mmc: "predicate `p'/1 has its predicate declaration combined
        // with a mode declaration, so it may not have a separate mode
        // declaration".
        messages_.Say(where,
            "%s has its modes in its %s declaration, so it may not have a separate mode declaration"_err_en_US,
            Text(*symbol), Keyword(pred.kind));
      }
      if (item.interface != symbol->exported) {
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
      if (item.interface == symbol->exported && symbol->exported &&
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
    result_.bySymbol_[symbol] = {result_.procedures_.size(), procedures.size()};
    result_.procedures_.insert(
        result_.procedures_.end(), procedures.begin(), procedures.end());
  }
}

std::unique_ptr<ModeChecking> CheckModes(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, const SemanticForm &,
    const NameResolution &resolution, TypeChecking &types,
    parser::Messages &messages, const ModeCheckOptions &options) {
  auto result{std::make_unique<ModeChecking>()};
  ModeChecker{*result, resolution, types, tokens, messages, options}.Run(tree);
  return result;
}

} // namespace merkurius::semantics
