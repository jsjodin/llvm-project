//===-- lib/Semantics/check-gpu-code.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/check-gpu-code.h"
#include "merkurius/Semantics/symbols.h"
#include "merkurius/Semantics/types.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include <optional>
#include <string>
#include <vector>

namespace merkurius::semantics {

using namespace parser::literals;

namespace {

bool IsNamed(const Type &type, llvm::StringRef module, llvm::StringRef name) {
  return type.kind == TypeKind::Named && type.constructor->module == module &&
      type.constructor->name == name;
}

// A primitive type whose values are one scalar.
bool IsScalar(const Type &type) {
  if (type.kind != TypeKind::Named || !type.constructor->module.empty()) {
    return false;
  }
  llvm::StringRef name{type.constructor->name};
  return name == "int" || name == "uint" || name == "int8" || name == "int16" ||
      name == "int32" || name == "int64" || name == "uint8" ||
      name == "uint16" || name == "uint32" || name == "uint64" ||
      name == "float" || name == "char" || name == "character";
}

bool IsString(const Type &type) {
  return type.kind == TypeKind::Named && type.constructor->module.empty() &&
      type.constructor->name == "string";
}

bool HasVariable(const Type &type) {
  if (type.kind == TypeKind::Variable || type.kind == TypeKind::Existential) {
    return true;
  }
  if (type.result && HasVariable(*type.result)) {
    return true;
  }
  for (const Type *argument : type.arguments) {
    if (HasVariable(*argument)) {
      return true;
    }
  }
  return false;
}

// Where code of the module runs on a GPU: called from a target region; in
// a GPU process's code, on the process's lane; in the body of a distribute
// loop of a GPU process, on a worker team's first wave; and in the body of
// a loop that runs across a team's threads, a parallel do's or a distribute
// parallel do's. Each body is checked once in each.
enum class Context : unsigned { Region, Process, Team, Thread };

class GpuCodeChecker {
public:
  GpuCodeChecker(const parser::TokenSequence &tokens, const ModeChecking &modes,
      const DeterminismChecking &determinism, InstTypes &constructors,
      parser::Messages &messages)
      : tokens_{tokens}, modes_{modes}, determinism_{determinism},
        constructors_{constructors}, messages_{messages} {}

  void Check(const parser::ParseTree &tree);

private:
  parser::CharBlock Source(TokenRange range) const {
    return {tokens_.tokens[range.first].spelling.begin(),
        tokens_.tokens[range.last].spelling.end()};
  }
  // Why a value of the type is not flat, or none if it is. A type variable
  // is flat here: the type it stands for is checked where it is known.
  std::optional<std::string> NotFlat(const Type &);
  // Why a value of the type is not of a fixed size and flat, or none.
  std::optional<std::string> NotFixed(
      const Type &, llvm::SmallVectorImpl<const Type *> &enclosing);
  bool IsEnumeration(const Type &);

  // A body of the module's that runs on a GPU, in the context.
  void Visit(const ProcedureBody &, Context);
  void Walk(const ProcedureBody &, const Body &, Context);
  void CheckValue(const ProcedureBody &, const Body &, BodyVar);
  void CheckMessage(TokenRange where, const Type *message);
  void Call(
      const ProcedureBody &, const Body &, const Procedure *callee, Context);
  void CheckProcessLoops();
  bool Visited(const ProcedureBody &body, Context context) const {
    return visited_.contains({&body, static_cast<unsigned>(context)});
  }

  const parser::TokenSequence &tokens_;
  const ModeChecking &modes_;
  const DeterminismChecking &determinism_;
  InstTypes &constructors_;
  parser::Messages &messages_;
  // By body, the contexts it was visited in.
  llvm::DenseSet<std::pair<const ProcedureBody *, unsigned>> visited_;
  // The variables already checked, by body, so each is reported once.
  llvm::DenseSet<std::pair<const ProcedureBody *, BodyVar>> checked_;
  std::vector<std::pair<const ProcedureBody *, Context>> pending_;
  // Where a message was reported, so that a goal reached twice, as in the
  // bodies of two modes of one procedure, is reported once.
  llvm::DenseSet<const char *> reported_;
  // The mailboxes of the GPU kinds' processes themselves, by body.
  llvm::DenseSet<std::pair<const ProcedureBody *, BodyVar>> selves_;
};

bool GpuCodeChecker::IsEnumeration(const Type &type) {
  std::vector<ConstructorTypes> constructors;
  if (!constructors_.Constructors(&type, constructors)) {
    return false;
  }
  for (const ConstructorTypes &constructor : constructors) {
    if (!constructor.arguments.empty()) {
      return false;
    }
  }
  return true;
}

std::optional<std::string> GpuCodeChecker::NotFixed(
    const Type &type, llvm::SmallVectorImpl<const Type *> &enclosing) {
  if (type.kind == TypeKind::Variable) {
    return std::nullopt;
  }
  if (type.kind == TypeKind::Existential) {
    return "it has an existentially quantified type";
  }
  if (type.kind == TypeKind::HigherOrder) {
    return std::string{TypeToString(type)} + " is a higher-order type";
  }
  if (IsScalar(type) || IsNamed(type, "process", "mailbox")) {
    return std::nullopt;
  }
  if (IsString(type) || IsNamed(type, "array", "array")) {
    return std::string{TypeToString(type)} +
        " has no fixed size, which only a whole message may have";
  }
  if (type.kind == TypeKind::Tuple) {
    for (const Type *element : type.arguments) {
      if (auto why{NotFixed(*element, enclosing)}) {
        return why;
      }
    }
    return std::nullopt;
  }
  for (const Type *outer : enclosing) {
    if (outer->constructor == type.constructor) {
      return std::string{TypeToString(type)} + " is recursive";
    }
  }
  std::vector<ConstructorTypes> constructors;
  if (!constructors_.Constructors(&type, constructors)) {
    return std::string{TypeToString(type)} +
        " is abstract or primitive here, so its representation is unknown";
  }
  enclosing.push_back(&type);
  std::optional<std::string> why;
  for (const ConstructorTypes &constructor : constructors) {
    for (const Type *argument : constructor.arguments) {
      // An existentially quantified variable of the constructor: the type
      // has no variable of its own once it is ground.
      if (!HasVariable(type) && HasVariable(*argument)) {
        why = std::string{TypeToString(type)} +
            " has an existentially typed constructor";
      } else {
        why = NotFixed(*argument, enclosing);
      }
      if (why) {
        break;
      }
    }
    if (why) {
      break;
    }
  }
  enclosing.pop_back();
  return why;
}

std::optional<std::string> GpuCodeChecker::NotFlat(const Type &type) {
  if (IsString(type)) {
    return std::nullopt;
  }
  // What a receive gives: a whole message, or a timeout.
  if (IsNamed(type, "process", "received")) {
    return NotFlat(*type.arguments.front());
  }
  if (IsNamed(type, "array", "array")) {
    const Type &element{*type.arguments.front()};
    if (element.kind == TypeKind::Variable || IsScalar(element) ||
        IsEnumeration(element)) {
      return std::nullopt;
    }
    // Temporary (docs/INPROGRESS.md, GPU processes without omp target, step
    // 5, "Flat records", step 2): arrays of flat records.
    return std::string{TypeToString(type)} +
        "'s elements are not scalars or enumerations";
  }
  llvm::SmallVector<const Type *, 4> enclosing;
  return NotFixed(type, enclosing);
}

void GpuCodeChecker::CheckValue(
    const ProcedureBody &body, const Body &goal, BodyVar var) {
  if (!checked_.insert({&body, var}).second) {
    return;
  }
  const Type *type{body.vars[var].type};
  if (!type || IsNamed(*type, "io", "state")) {
    return;
  }
  if (std::optional<std::string> why{NotFlat(*type)};
      why && reported_.insert(Source(goal.tokens).begin()).second) {
    // A variable that the compiler made is named as a value.
    llvm::StringRef name{body.vars[var].name};
    std::string what{name.starts_with("V_") || name.starts_with("HeadVar__") ||
                name.starts_with("STATE_VARIABLE_")
            ? std::string{"a value"}
            : name.str()};
    messages_.Say(Source(goal.tokens),
        "code of a process kind that runs on a GPU must make only flat values, but this makes %s of type %s: %s"_err_en_US,
        what, TypeToString(*type), *why);
  }
}

void GpuCodeChecker::CheckMessage(TokenRange where, const Type *message) {
  if (!message) {
    return;
  }
  if (std::optional<std::string> why{NotFlat(*message)};
      why && reported_.insert(Source(where).begin()).second) {
    messages_.Say(Source(where),
        "a message that code running on a GPU sends or receives must be flat, but this one is of type %s: %s"_err_en_US,
        TypeToString(*message), *why);
  }
}

void GpuCodeChecker::Call(const ProcedureBody &body, const Body &goal,
    const Procedure *callee, Context context) {
  bool process{context != Context::Region};
  if (!callee) {
    return;
  }
  const Symbol &symbol{*callee->symbol};
  bool send{
      symbol.module == "process" && symbol.name == "send" && symbol.arity == 4};
  bool receive{symbol.module == "process" && symbol.name == "receive" &&
      symbol.arity == 6};
  bool any{symbol.module == "process" && symbol.name == "any"};
  if (send && goal.arguments.size() >= 2) {
    CheckMessage(goal.tokens, body.vars[goal.arguments[1]].type);
  }
  // A kind's own mailbox's messages are checked at the kind's declaration.
  if (receive && goal.arguments.size() >= 4 &&
      !selves_.contains({&body, goal.arguments[0]})) {
    // received(T): its argument is the message's type.
    const Type *received{body.vars[goal.arguments[3]].type};
    if (received && received->arguments.size() == 1) {
      CheckMessage(goal.tokens, received->arguments.front());
    }
  }
  const ProcedureBody *called{determinism_.BodyOf(*callee)};
  // The values that a procedure of the module makes are checked in its
  // body, where they are made.
  if (process && !called) {
    for (auto [i, var] : llvm::enumerate(goal.arguments)) {
      // any's closure, the receive's selector, which code generation
      // removes (MLIRGenOpenMP.cpp, ReceiveInTarget).
      if ((receive && i == 1) || any) {
        checked_.insert({&body, var});
        continue;
      }
      CheckValue(body, goal, var);
    }
    if (goal.kind == BodyKind::Unify) {
      if (any) {
        checked_.insert({&body, goal.left});
      } else {
        CheckValue(body, goal, goal.left);
      }
    }
  }
  if (called) {
    if (!Visited(*called, context)) {
      pending_.push_back({called, context});
    }
  }
}

// The context of a GPU process's code under the loop directive, which its
// calls run in: a distribute loop's body on a team's first wave, and a
// parallel do's or a distribute parallel do's across a team's threads.
static Context InnerContext(Context context, llvm::omp::Directive id) {
  if (context == Context::Region) {
    return context;
  }
  if (id == llvm::omp::OMPD_distribute) {
    return Context::Team;
  }
  if (id == llvm::omp::OMPD_distribute_parallel_do ||
      id == llvm::omp::OMPD_parallel_do) {
    return Context::Thread;
  }
  return context;
}

void GpuCodeChecker::Walk(
    const ProcedureBody &body, const Body &goal, Context context) {
  bool process{context != Context::Region};
  switch (goal.kind) {
  case BodyKind::Unify:
    if (goal.unify == UnifyKind::Call) {
      Call(body, goal, goal.modes ? goal.modes->callee : nullptr, context);
    } else if (process && goal.unify == UnifyKind::Construct) {
      CheckValue(body, goal, goal.left);
    }
    if (goal.lambda) {
      Walk(body, *goal.lambda, context);
    }
    break;
  case BodyKind::Call:
    Call(body, goal, goal.modes ? goal.modes->callee : nullptr, context);
    break;
  case BodyKind::HOCall:
    if (process) {
      CheckValue(body, goal, goal.closure);
    }
    break;
  case BodyKind::Switch:
    for (const SwitchCase &arm : goal.cases) {
      Walk(body, *arm.goal, context);
    }
    break;
  default: {
    Context inner{goal.kind == BodyKind::Scope && goal.directive
            ? InnerContext(context, goal.directive->id)
            : context};
    for (const Body *g : goal.goals) {
      Walk(body, *g, inner);
    }
    break;
  }
  }
}

// The variables of a body that its sends and receives cover: the
// receives' selectors, any's closures, which code generation removes
// (MLIRGenOpenMP.cpp, ReceiveInTarget), so that the process makes none;
// and the messages sent and received, whose types are checked as messages.
static void FindCovered(
    const Body &goal, llvm::SmallVectorImpl<BodyVar> &covered) {
  if (goal.kind == BodyKind::Call && goal.modes && goal.modes->callee) {
    const Symbol &symbol{*goal.modes->callee->symbol};
    if (symbol.module == "process" && symbol.name == "receive" &&
        symbol.arity == 6 && goal.arguments.size() >= 4) {
      covered.push_back(goal.arguments[1]);
      covered.push_back(goal.arguments[3]);
    }
    if (symbol.module == "process" && symbol.name == "send" &&
        symbol.arity == 4 && goal.arguments.size() >= 2) {
      covered.push_back(goal.arguments[1]);
    }
  }
  if (goal.lambda) {
    FindCovered(*goal.lambda, covered);
  }
  for (const SwitchCase &arm : goal.cases) {
    FindCovered(*arm.goal, covered);
  }
  for (const Body *inner : goal.goals) {
    FindCovered(*inner, covered);
  }
}

void GpuCodeChecker::Visit(const ProcedureBody &body, Context context) {
  if (!visited_.insert({&body, static_cast<unsigned>(context)}).second ||
      !body.body) {
    return;
  }
  if (context != Context::Region) {
    llvm::SmallVector<BodyVar> covered;
    FindCovered(*body.body, covered);
    for (BodyVar var : covered) {
      checked_.insert({&body, var});
    }
  }
  Walk(body, *body.body, context);
}

// The target regions of a body, and what they call: their messages.
void FindRegions(const Body &goal, llvm::SmallVectorImpl<const Body *> &out) {
  if (goal.kind == BodyKind::Scope && goal.directive) {
    llvm::ArrayRef<llvm::omp::Directive> leaves{
        llvm::omp::getLeafConstructsOrSelf(goal.directive->id)};
    if (!leaves.empty() && leaves.front() == llvm::omp::OMPD_target) {
      out.push_back(&goal);
      return;
    }
  }
  if (goal.lambda) {
    FindRegions(*goal.lambda, out);
  }
  for (const SwitchCase &arm : goal.cases) {
    FindRegions(*arm.goal, out);
  }
  for (const Body *inner : goal.goals) {
    FindRegions(*inner, out);
  }
}

void GpuCodeChecker::Check(const parser::ParseTree &tree) {
  for (const parser::Item &item : tree.items) {
    const auto *decl{std::get_if<const parser::ProcessDecl *>(&item)};
    if (!decl || !(*decl)->onGpu) {
      continue;
    }
    for (const Procedure &procedure : modes_.procedures()) {
      const auto *declared{std::get_if<const parser::PredDecl *>(
          &procedure.symbol->declaration)};
      if (!declared || *declared != (*decl)->kindDecl) {
        continue;
      }
      if (const ProcedureBody *body{determinism_.BodyOf(procedure)}) {
        // What its own mailbox, before the I/O states, receives is its
        // messages.
        if (body->head.size() >= 3) {
          BodyVar self{body->head[body->head.size() - 3]};
          const Type *type{body->vars[self].type};
          if (type && IsNamed(*type, "process", "mailbox")) {
            CheckMessage((*decl)->tokens, type->arguments.front());
            selves_.insert({body, self});
          }
        }
        pending_.push_back({body, Context::Process});
      }
    }
  }
  for (const ProcedureBody &body : determinism_.bodies()) {
    if (!body.body) {
      continue;
    }
    llvm::SmallVector<const Body *> regions;
    FindRegions(*body.body, regions);
    for (const Body *region : regions) {
      Walk(body, *region, Context::Region);
    }
  }
  while (!pending_.empty()) {
    auto [body, context] = pending_.back();
    pending_.pop_back();
    Visit(*body, context);
  }
  CheckProcessLoops();
}

// The goals of a body under a directive, each with whether it is in a target
// region.
void FindDirectives(const Body &goal, bool inRegion,
    llvm::SmallVectorImpl<std::pair<const Body *, bool>> &out) {
  if (goal.kind == BodyKind::Scope && goal.directive) {
    out.push_back({&goal, inRegion});
    llvm::ArrayRef<llvm::omp::Directive> leaves{
        llvm::omp::getLeafConstructsOrSelf(goal.directive->id)};
    inRegion = inRegion ||
        (!leaves.empty() && leaves.front() == llvm::omp::OMPD_target);
  }
  if (goal.lambda) {
    FindDirectives(*goal.lambda, inRegion, out);
  }
  for (const SwitchCase &arm : goal.cases) {
    FindDirectives(*arm.goal, inRegion, out);
  }
  for (const Body *inner : goal.goals) {
    FindDirectives(*inner, inRegion, out);
  }
}

// The loops of a GPU process (docs/INPROGRESS.md, GPU processes that run
// loops): under distribute parallel do, a loop runs across the process's
// teams' threads; under distribute, across its teams, a worker team's first
// wave running an iteration's body, in which a loop under parallel do runs
// across the team's threads. Their code is that of a process kind that runs
// on a GPU. In a target region a distribute loop would be OpenMP's, under
// target teams, which is not implemented. On the host, as in a kind spawned
// there, distribute loops are parallel do loops. A GPU process's code has no
// other directive, as its kernel is compiled without OpenMP.
void GpuCodeChecker::CheckProcessLoops() {
  for (const ProcedureBody &body : determinism_.bodies()) {
    if (!body.body) {
      continue;
    }
    bool region{Visited(body, Context::Region)};
    bool process{Visited(body, Context::Process)};
    bool team{Visited(body, Context::Team)};
    bool thread{Visited(body, Context::Thread)};
    llvm::SmallVector<std::pair<const Body *, bool>> directives;
    FindDirectives(*body.body, /*inRegion=*/false, directives);
    for (auto [directive, inRegion] : directives) {
      parser::CharBlock where{Source(directive->tokens)};
      llvm::omp::Directive id{directive->directive->id};
      std::string name{parser::EnumToString(id)};
      bool distribute{id == llvm::omp::OMPD_distribute_parallel_do ||
          id == llvm::omp::OMPD_distribute};
      if (!distribute && id != llvm::omp::OMPD_parallel_do) {
        if (!inRegion && (process || team || thread) &&
            reported_.insert(where.begin()).second) {
          messages_.Say(where,
              "a process kind that runs on a GPU runs its loops across its "
              "teams under distribute parallel do, or under distribute with "
              "parallel do loops in their bodies, and its code has no %s "
              "directive"_err_en_US,
              name);
        }
        continue;
      }
      if (!distribute) {
        // A parallel do of a region's code, or of the host's, is OpenMP's.
        if (inRegion || (!process && !thread) ||
            !reported_.insert(where.begin()).second) {
          continue;
        }
        if (process) {
          messages_.Say(where,
              "a parallel do loop in a GPU process's code runs across a "
              "team's threads, so it must be in the body of a distribute "
              "loop, or in a predicate of the module that such a body "
              "calls"_err_en_US);
        } else {
          messages_.Say(where,
              "a parallel do loop in a GPU process's code cannot be in the "
              "body of another loop that runs across a team's "
              "threads"_err_en_US);
        }
        continue;
      }
      if (!reported_.insert(where.begin()).second) {
        continue;
      }
      if (inRegion || region) {
        // OpenMP's distribute under target teams, which differs.
        messages_.Say(where,
            "not yet implemented: a %s loop in a target region; in a GPU "
            "process's code it runs across the process's teams"_err_en_US,
            name);
      } else if (team || thread) {
        messages_.Say(where,
            "a %s loop runs across the teams of a GPU process, so it cannot "
            "be in the body of another loop"_err_en_US,
            name);
      } else if (!process) {
        messages_.Say(where,
            "a %s loop runs across the teams of a GPU process, so it must be "
            "in the code of a process kind declared on gpu, or in a "
            "predicate of the module that such a kind calls"_err_en_US,
            name);
      }
    }
  }
}

} // namespace

void CheckGpuCode(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, const ModeChecking &modes,
    const DeterminismChecking &determinism, InstTypes &constructors,
    parser::Messages &messages) {
  GpuCodeChecker{tokens, modes, determinism, constructors, messages}.Check(
      tree);
}

} // namespace merkurius::semantics
