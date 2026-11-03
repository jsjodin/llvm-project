//===-- lib/CodeGen/Decomposer.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Decomposer.h"
#include "merkurius/Common/idioms.h"
#include "merkurius/Common/visit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Frontend/OpenMP/ClauseT.h"
#include "llvm/Frontend/OpenMP/ConstructDecompositionT.h"
#include <cassert>
#include <utility>

namespace merkurius::codegen::omp {

// tomp's type parameters: an object is a variable, by the number of its
// name in the directive, and an expression is a data-term. Types are not
// used, but must differ from the others, which overloads tell apart.
struct NoType {};
using TypeTy = const NoType *;
using IdTy = unsigned;
using ExprTy = const parser::DataTerm *;

} // namespace merkurius::codegen::omp

namespace tomp::type {
template <>
struct ObjectT<merkurius::codegen::omp::IdTy,
    merkurius::codegen::omp::ExprTy> {
  using IdTy = merkurius::codegen::omp::IdTy;
  using ExprTy = merkurius::codegen::omp::ExprTy;

  IdTy id() const { return identity; }
  const std::optional<ExprTy> &ref() const { return designator; }

  bool operator<(const ObjectT &other) const {
    return identity < other.identity;
  }

  IdTy identity{0};
  std::optional<ExprTy> designator;
};
} // namespace tomp::type

namespace merkurius::codegen::omp {
namespace {

using Object = tomp::ObjectT<IdTy, ExprTy>;
using ObjectList = tomp::ObjectListT<IdTy, ExprTy>;
using ClauseBase = tomp::ClauseT<TypeTy, IdTy, ExprTy>;

// A tomp clause that remembers the clause of the directive it came from;
// one the decomposition adds has none.
struct Clause : public ClauseBase {
  Clause() = default;
  Clause(ClauseBase &&base) : ClauseBase(std::move(base)) {}
  const parser::OmpClause *original{nullptr};
};

// Clause kinds with an applyClause() of their own, which selects it by the
// payload's type, or whose contents the algorithm reads.
bool NeedsSpecificHandling(llvm::omp::Clause kind) {
  switch (kind) {
  case llvm::omp::OMPC_allocate:
  case llvm::omp::OMPC_collapse:
  case llvm::omp::OMPC_default:
  case llvm::omp::OMPC_dyn_groupprivate:
  case llvm::omp::OMPC_firstprivate:
  case llvm::omp::OMPC_if:
  case llvm::omp::OMPC_lastprivate:
  case llvm::omp::OMPC_linear:
  case llvm::omp::OMPC_map:
  case llvm::omp::OMPC_nowait:
  case llvm::omp::OMPC_ompx_attribute:
  case llvm::omp::OMPC_ompx_bare:
  case llvm::omp::OMPC_order:
  case llvm::omp::OMPC_private:
  case llvm::omp::OMPC_reduction:
  case llvm::omp::OMPC_shared:
  case llvm::omp::OMPC_thread_limit:
    return true;
  default:
    return false;
  }
}

class Converter {
public:
  // The number of a variable's name, the same for each occurrence.
  IdTy Id(llvm::StringRef name) {
    return names_.try_emplace(name, names_.size() + 1).first->second;
  }

  Clause Convert(const parser::OmpClause &clause) {
    namespace tc = tomp::clause;
    llvm::omp::Clause kind{ClauseId(clause)};
    return common::visit(
        common::visitors{
            [&](const parser::OmpReductionClause *c) {
              ObjectList list;
              for (const parser::Variable *v : c->variables) {
                list.push_back(Object{Id(v->name), std::nullopt});
              }
              return Make(kind,
                  tc::ReductionT<TypeTy, IdTy, ExprTy>{
                      {/*ReductionModifier=*/std::nullopt,
                          /*ReductionIdentifiers=*/{}, /*List=*/list}},
                  clause);
            },
            [&](const parser::OmpMapClause *c) {
              ObjectList list;
              for (const parser::OmpMapObject *object : c->objects) {
                list.push_back(Object{Id(object->variable->name), std::nullopt});
              }
              return Make(kind,
                  tc::MapT<TypeTy, IdTy, ExprTy>{
                      {/*MapType=*/std::nullopt,
                          /*MapTypeModifiers=*/std::nullopt,
                          /*AttachModifier=*/std::nullopt,
                          /*RefModifier=*/std::nullopt,
                          /*Mappers=*/std::nullopt, /*Iterator=*/std::nullopt,
                          /*LocatorList=*/list}},
                  clause);
            },
            [&](const parser::OmpGenericClause *c) {
              return Generic(kind, *c, clause);
            },
            [&](const auto *) { return Plain(kind, clause); },
        },
        clause);
  }

private:
  template <typename Specific>
  static Clause Make(llvm::omp::Clause id, Specific &&specific,
      const parser::OmpClause &original) {
    Clause c{ClauseBase{id, std::forward<Specific>(specific)}};
    c.original = &original;
    return c;
  }

  // A clause by its kind alone, with an empty payload that goes through
  // the algorithm's generic applyClause(), which reads only the kind.
  static Clause Plain(llvm::omp::Clause id, const parser::OmpClause &original) {
    assert(!NeedsSpecificHandling(id) &&
        "the clause needs a payload for the decomposition");
    return Make(id, tomp::clause::ThreadsT<TypeTy, IdTy, ExprTy>{}, original);
  }

  // The variables among a clause's arguments.
  ObjectList Objects(llvm::ArrayRef<parser::DataTerm> arguments) {
    ObjectList list;
    for (const parser::DataTerm &argument : arguments) {
      if (const auto *v{std::get_if<const parser::Variable *>(&argument)}) {
        list.push_back(Object{Id((*v)->name), &argument});
      } else if (const auto *s{
                     std::get_if<const parser::StateVar *>(&argument)}) {
        list.push_back(Object{Id((*s)->name), &argument});
      }
    }
    return list;
  }

  Clause Generic(llvm::omp::Clause kind, const parser::OmpGenericClause &c,
      const parser::OmpClause &original) {
    namespace tc = tomp::clause;
    switch (kind) {
    case llvm::omp::OMPC_firstprivate:
      return Make(kind,
          tc::FirstprivateT<TypeTy, IdTy, ExprTy>{Objects(c.arguments)},
          original);
    case llvm::omp::OMPC_private:
      return Make(kind,
          tc::PrivateT<TypeTy, IdTy, ExprTy>{Objects(c.arguments)}, original);
    case llvm::omp::OMPC_shared:
      return Make(kind,
          tc::SharedT<TypeTy, IdTy, ExprTy>{Objects(c.arguments)}, original);
    case llvm::omp::OMPC_lastprivate:
      return Make(kind,
          tc::LastprivateT<TypeTy, IdTy, ExprTy>{
              {/*LastprivateModifier=*/std::nullopt,
                  /*List=*/Objects(c.arguments)}},
          original);
    case llvm::omp::OMPC_linear:
      return Make(kind,
          tc::LinearT<TypeTy, IdTy, ExprTy>{
              {/*StepComplexModifier=*/std::nullopt,
                  /*LinearModifier=*/std::nullopt,
                  /*List=*/Objects(c.arguments)}},
          original);
    case llvm::omp::OMPC_if: {
      // if(parallel : C) names the leaf it applies to.
      std::optional<llvm::omp::Directive> modifier;
      if (c.modifiers.size() == 1) {
        if (const auto *f{std::get_if<const parser::FunctorData *>(
                &c.modifiers.front())};
            f && (*f)->arguments.empty()) {
          llvm::omp::Directive id{llvm::omp::getOpenMPDirectiveKind(
              (*f)->name->name)};
          if (id != llvm::omp::Directive::OMPD_unknown) {
            modifier = id;
          }
        }
      }
      ExprTy condition{c.arguments.empty() ? nullptr : &c.arguments.front()};
      return Make(kind,
          tc::IfT<TypeTy, IdTy, ExprTy>{
              {/*DirectiveNameModifier=*/modifier,
                  /*IfExpression=*/condition}},
          original);
    }
    // Clauses with an applyClause() of their own whose contents the
    // algorithm does not read: the payload's type selects it.
    case llvm::omp::OMPC_allocate:
      return Make(kind,
          tc::AllocateT<TypeTy, IdTy, ExprTy>{
              {std::nullopt, std::nullopt, /*List=*/{}}},
          original);
    case llvm::omp::OMPC_collapse:
      return Make(
          kind, tc::CollapseT<TypeTy, IdTy, ExprTy>{/*N=*/nullptr}, original);
    case llvm::omp::OMPC_default:
      return Make(kind,
          tc::DefaultT<TypeTy, IdTy, ExprTy>{
              tc::DefaultT<TypeTy, IdTy, ExprTy>::DataSharingAttribute::Shared},
          original);
    case llvm::omp::OMPC_dyn_groupprivate:
      return Make(kind,
          tc::DynGroupprivateT<TypeTy, IdTy, ExprTy>{
              {std::nullopt, std::nullopt, /*Size=*/nullptr}},
          original);
    case llvm::omp::OMPC_nowait:
      return Make(kind, tc::NowaitT<TypeTy, IdTy, ExprTy>{}, original);
    case llvm::omp::OMPC_ompx_attribute:
      return Make(kind, tc::OmpxAttributeT<TypeTy, IdTy, ExprTy>{}, original);
    case llvm::omp::OMPC_ompx_bare:
      return Make(kind, tc::OmpxBareT<TypeTy, IdTy, ExprTy>{}, original);
    case llvm::omp::OMPC_order:
      return Make(kind,
          tc::OrderT<TypeTy, IdTy, ExprTy>{
              {std::nullopt,
                  tc::OrderT<TypeTy, IdTy, ExprTy>::Ordering::Concurrent}},
          original);
    case llvm::omp::OMPC_thread_limit:
      return Make(kind, tc::ThreadLimitT<TypeTy, IdTy, ExprTy>{/*List=*/{}},
          original);
    default:
      return Plain(kind, original);
    }
  }

  llvm::StringMap<IdTy> names_;
};

// What ConstructDecompositionT asks of its user.
struct Helper {
  // An object is a variable, its own base.
  std::optional<Object> getBaseObject(const Object &object) const {
    return object;
  }
  std::optional<Object> getLoopIterVar() const { return counter; }
  bool isClauseAllowedOnDirective(llvm::omp::Clause clause,
      llvm::omp::Directive directive, llvm::omp::Version version) const {
    return llvm::omp::isAllowedClauseForDirective(directive, clause, version);
  }
  std::optional<Object> counter;
};

} // namespace

llvm::omp::Clause ClauseId(const parser::OmpClause &clause) {
  return common::visit(
      common::visitors{
          [](const parser::OmpScheduleClause *) {
            return llvm::omp::OMPC_schedule;
          },
          [](const parser::OmpReductionClause *) {
            return llvm::omp::OMPC_reduction;
          },
          [](const parser::OmpMapClause *) { return llvm::omp::OMPC_map; },
          [](const parser::OmpIndexClause *) {
            return llvm::omp::OMPC_unknown;
          },
          [](const parser::OmpUpperClause *) {
            return llvm::omp::OMPC_unknown;
          },
          [](const parser::OmpGenericClause *c) { return c->id; },
      },
      clause);
}

Decomposition Decompose(unsigned version,
    const parser::OmpDirective &directive,
    std::optional<llvm::StringRef> counter) {
  Converter converter;
  Helper helper;
  if (counter) {
    helper.counter = Object{converter.Id(*counter), std::nullopt};
  }
  llvm::SmallVector<Clause> input;
  for (const parser::OmpClause &clause : directive.clauses) {
    if (ClauseId(clause) != llvm::omp::OMPC_unknown) {
      input.push_back(converter.Convert(clause));
    }
  }
  tomp::ConstructDecompositionT<Clause, Helper> decomposition(
      llvm::omp::Version(version), helper, directive.id,
      llvm::ArrayRef<Clause>(input));
  Decomposition result;
  if (decomposition.output.empty()) {
    // A clause that no leaf allows.
    llvm::ArrayRef<llvm::omp::Directive> leaves{
        llvm::omp::getLeafConstructsOrSelf(directive.id)};
    for (const Clause &clause : input) {
      if (llvm::none_of(leaves, [&](llvm::omp::Directive leaf) {
            return helper.isClauseAllowedOnDirective(
                clause.id, leaf, llvm::omp::Version(version));
          })) {
        result.misplaced = clause.original;
        break;
      }
    }
    if (!result.misplaced && !input.empty()) {
      result.misplaced = input.front().original;
    }
    return result;
  }
  for (const tomp::DirectiveWithClauses<Clause> &leaf :
      decomposition.output) {
    Leaf out;
    out.id = leaf.id;
    for (const Clause &clause : leaf.clauses) {
      if (clause.original) {
        out.clauses.push_back(clause.original);
        continue;
      }
      // An if whose modifier named this leaf, which the decomposition
      // copies without the modifier, with the same condition.
      if (clause.id == llvm::omp::OMPC_if) {
        const auto &ifClause{
            std::get<tomp::clause::IfT<TypeTy, IdTy, ExprTy>>(clause.u)};
        ExprTy condition{std::get<ExprTy>(ifClause.t)};
        const auto *original{llvm::find_if(input, [&](const Clause &c) {
          if (c.id != llvm::omp::OMPC_if) {
            return false;
          }
          const auto &i{std::get<tomp::clause::IfT<TypeTy, IdTy, ExprTy>>(c.u)};
          return std::get<ExprTy>(i.t) == condition;
        })};
        if (original != input.end()) {
          out.clauses.push_back(original->original);
          continue;
        }
      }
      out.synthesized.push_back(clause.id);
    }
    result.queue.push_back(std::move(out));
  }
  return result;
}

} // namespace merkurius::codegen::omp
