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

  // A body of the module's that runs on a GPU, in a process kind's code if
  // `process`, else called from a target region: each is checked once.
  void Visit(const ProcedureBody &, bool process);
  void Walk(const ProcedureBody &, const Body &, bool process);
  void CheckValue(const ProcedureBody &, const Body &, BodyVar);
  void CheckMessage(TokenRange where, const Type *message);
  void Call(const ProcedureBody &, const Body &, const Procedure *callee,
      bool process);

  const parser::TokenSequence &tokens_;
  const ModeChecking &modes_;
  const DeterminismChecking &determinism_;
  InstTypes &constructors_;
  parser::Messages &messages_;
  // By body, whether visited as a process kind's code (1) or a region's (0).
  llvm::DenseSet<std::pair<const ProcedureBody *, unsigned>> visited_;
  // The variables already checked, by body, so each is reported once.
  llvm::DenseSet<std::pair<const ProcedureBody *, BodyVar>> checked_;
  std::vector<std::pair<const ProcedureBody *, bool>> pending_;
  // Where a message was reported, so that a goal reached twice, as in the
  // bodies of two modes of one procedure, is reported once.
  llvm::DenseSet<const char *> reported_;
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
    messages_.Say(Source(goal.tokens),
        "code of a process kind that runs on a GPU must make only flat values, but this makes %s of type %s: %s"_err_en_US,
        body.vars[var].name, TypeToString(*type), *why);
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
    const Procedure *callee, bool process) {
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
  if (receive && goal.arguments.size() >= 4) {
    // received(T): its argument is the message's type.
    const Type *received{body.vars[goal.arguments[3]].type};
    if (received && received->arguments.size() == 1) {
      CheckMessage(goal.tokens, received->arguments.front());
    }
  }
  if (process) {
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
  if (const ProcedureBody *called{determinism_.BodyOf(*callee)}) {
    if (!visited_.contains({called, process ? 1u : 0u})) {
      pending_.push_back({called, process});
    }
  }
}

void GpuCodeChecker::Walk(
    const ProcedureBody &body, const Body &goal, bool process) {
  switch (goal.kind) {
  case BodyKind::Unify:
    if (goal.unify == UnifyKind::Call) {
      Call(body, goal, goal.modes ? goal.modes->callee : nullptr, process);
    } else if (process && goal.unify == UnifyKind::Construct) {
      CheckValue(body, goal, goal.left);
    }
    if (goal.lambda) {
      Walk(body, *goal.lambda, process);
    }
    break;
  case BodyKind::Call:
    Call(body, goal, goal.modes ? goal.modes->callee : nullptr, process);
    break;
  case BodyKind::HOCall:
    if (process) {
      CheckValue(body, goal, goal.closure);
    }
    break;
  case BodyKind::Switch:
    for (const SwitchCase &arm : goal.cases) {
      Walk(body, *arm.goal, process);
    }
    break;
  default:
    for (const Body *inner : goal.goals) {
      Walk(body, *inner, process);
    }
    break;
  }
}

// The selectors of a body's receives, any's closures, which code generation
// removes (MLIRGenOpenMP.cpp, ReceiveInTarget), so the process makes none.
static void FindSelectors(
    const Body &goal, llvm::SmallVectorImpl<BodyVar> &selectors) {
  if (goal.kind == BodyKind::Call && goal.modes && goal.modes->callee) {
    const Symbol &symbol{*goal.modes->callee->symbol};
    if (symbol.module == "process" && symbol.name == "receive" &&
        symbol.arity == 6 && goal.arguments.size() >= 2) {
      selectors.push_back(goal.arguments[1]);
    }
  }
  if (goal.lambda) {
    FindSelectors(*goal.lambda, selectors);
  }
  for (const SwitchCase &arm : goal.cases) {
    FindSelectors(*arm.goal, selectors);
  }
  for (const Body *inner : goal.goals) {
    FindSelectors(*inner, selectors);
  }
}

void GpuCodeChecker::Visit(const ProcedureBody &body, bool process) {
  if (!visited_.insert({&body, process ? 1u : 0u}).second || !body.body) {
    return;
  }
  if (process) {
    llvm::SmallVector<BodyVar> selectors;
    FindSelectors(*body.body, selectors);
    for (BodyVar var : selectors) {
      checked_.insert({&body, var});
    }
  }
  Walk(body, *body.body, process);
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
        // What its mailboxes receive is its messages.
        for (BodyVar var : body->head) {
          const Type *type{body->vars[var].type};
          if (type && IsNamed(*type, "process", "mailbox")) {
            CheckMessage((*decl)->tokens, type->arguments.front());
          }
        }
        pending_.push_back({body, true});
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
      Walk(body, *region, /*process=*/false);
    }
  }
  while (!pending_.empty()) {
    auto [body, process] = pending_.back();
    pending_.pop_back();
    Visit(*body, process);
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
