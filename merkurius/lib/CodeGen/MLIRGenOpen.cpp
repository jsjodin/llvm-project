//===-- lib/CodeGen/MLIRGenOpen.cpp - Open terms --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Open terms (design document, Language extensions, "Open terms";
// docs/INPROGRESS.md, Open types, step 4). A value of open(T) is a word of
// T's open layout, which the runtime's generic walkers take with T's
// type_info (runtime/open.h): full unification calls merk_open_unify, a
// fresh variable is merk_open_new_var's cell, and a procedure that can bind
// open variables marks the trail at each goal that catches failure
// (MLIRGenControl.cpp).
//
//===----------------------------------------------------------------------===//

#include "MLIRGenImpl.h"

namespace merkurius::codegen::mlirgen {

bool MerkGenerator::IsOpenType(const semantics::Type *type) {
  return types_.types().IsOpen(type);
}

bool MerkGenerator::IsClosedType(const semantics::Type *type) {
  return types_.types().IsClosed(type);
}

bool MerkGenerator::MentionsOpenType(const semantics::Type *type) {
  return types_.types().MentionsOpen(type);
}

// Temporary (open types step 4(d), docs/INPROGRESS.md): by the procedure's
// variables, rather than by the goals with an `any` nonlocal. A variable
// whose inst has `any` in it may be bound, whatever its type says, such as
// one of a type variable that is open(U) at the call.
bool MerkGenerator::Trails(const ProcedureBody &body) {
  auto [it, inserted]{trails_.try_emplace(&body, false)};
  if (inserted) {
    it->second = llvm::any_of(body.vars, [&](const semantics::BodyVarInfo &v) {
      return v.any || MentionsOpenType(v.type);
    });
  }
  return it->second;
}

mlir::Value MerkGenerator::TrailMark(mlir::Location loc) {
  return merk::ForeignCallOp::create(builder_, loc,
      mlir::TypeRange{builder_.getI64Type()}, "merk_open_trail_mark",
      mlir::ValueRange{}, /*can_fail=*/false)
      .getResult(0);
}

void MerkGenerator::TrailUndo(mlir::Value mark, mlir::Location loc) {
  merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
      "merk_open_trail_undo", mlir::ValueRange{mark}, /*can_fail=*/false);
}

void MerkGenerator::TrailRelease(mlir::Value mark, mlir::Location loc) {
  merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
      "merk_open_trail_release", mlir::ValueRange{mark}, /*can_fail=*/false);
}

mlir::Value MerkGenerator::TrailMarks(mlir::Location loc) {
  return merk::ForeignCallOp::create(builder_, loc,
      mlir::TypeRange{builder_.getI64Type()}, "merk_open_trail_marks",
      mlir::ValueRange{}, /*can_fail=*/false)
      .getResult(0);
}

void MerkGenerator::FreshVariables(
    const Body &goal, const ProcedureBody &body) {
  mlir::Location loc{Location(goal.tokens)};
  for (BodyVar var : goal.fresh) {
    if (IsClosedType(body.vars[var].type)) {
      // Temporary (open types step 4(c), docs/INPROGRESS.md).
      NotYet(Where(goal.tokens), "closed(T) values that may be unbound");
      return;
    }
    values_[var] = merk::ForeignCallOp::create(builder_, loc,
        mlir::TypeRange{merk::WordType::get(&context_)}, "merk_open_new_var",
        mlir::ValueRange{}, /*can_fail=*/false)
                       .getResult(0);
  }
}

bool MerkGenerator::IsOpenUnify(const Body &goal, const ProcedureBody &body) {
  return goal.unify == UnifyKind::GetStructure ||
      goal.unify == UnifyKind::Unify ||
      (!goal.isVar && goal.unify != UnifyKind::Call &&
          IsOpenType(body.vars[goal.left].type));
}

void MerkGenerator::OpenUnify(const Body &goal, const ProcedureBody &body) {
  parser::CharBlock at{Where(goal.tokens)};
  const semantics::Type *type{body.vars[goal.left].type};
  // Temporary (open types step 4(b), docs/INPROGRESS.md): constructions and
  // get_structure in the open layout.
  if (goal.unify == UnifyKind::GetStructure) {
    NotYet(at, "get_structure on open terms");
    return;
  }
  if (!goal.isVar) {
    NotYet(at, "constructions and deconstructions of open terms");
    return;
  }
  // Full unification. A closed(T) that may be unbound is a box or a
  // variable, which its ordinary word is not, outside open(T).
  if (!IsOpenType(type) && MentionsClosedType(type)) {
    // Temporary (open types step 4(c), docs/INPROGRESS.md).
    NotYet(at, "closed(T) values that may be unbound");
    return;
  }
  Test(goal, ValueOf(goal.left), ValueOf(goal.var), type);
}

bool MerkGenerator::MentionsClosedType(const semantics::Type *type) {
  return type &&
      (IsClosedType(type) ||
          llvm::any_of(type->arguments,
              [&](const semantics::Type *t) { return MentionsClosedType(t); }));
}

void MerkGenerator::OpenEqual(mlir::Value x, mlir::Value y,
    const semantics::Type *semantic, mlir::Location loc) {
  mlir::Value info{TypeInfoOfType(semantic->arguments.front(), loc, equalAt_)};
  if (!info) {
    failed_ = true;
    return;
  }
  auto word{merk::WordType::get(&context_)};
  auto call{merk::ForeignCallOp::create(builder_, loc,
      mlir::TypeRange{builder_.getI1Type()}, "merk_open_unify",
      mlir::ValueRange{info, Coerce(x, word, loc), Coerce(y, word, loc)},
      /*can_fail=*/true)};
  merk::GuardOp::create(builder_, loc, call.getResult(0));
}

// An open(T)'s word is described by open(T)'s type_info, as a type
// variable's is by its own. That is a cell, which the runtime keeps for each
// T (merk_open_type_info), since making one on the heap would be a safepoint
// at which the value, a parameter, could already be live; a foreign call is
// none. It is made where T's type_info is: at the procedure's start for a
// ground T, after T's type_info for a type variable.
bool MerkGenerator::NoteOpenRoot(
    mlir::Value value, const semantics::Type *type) {
  const semantics::Type *argument{type->arguments.front()};
  mlir::Location loc{value.getLoc()};
  mlir::OpBuilder::InsertPoint here{builder_.saveInsertionPoint()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  llvm::SmallVector<unsigned> variables;
  mlir::Value argumentInfo;
  if (argument->kind == semantics::TypeKind::Variable) {
    argumentInfo = typeInfos_.lookup(argument->var);
    if (!argumentInfo) {
      argumentInfo = givenInfos_.lookup(argument->var);
    }
    if (!argumentInfo) {
      return false;
    }
    if (mlir::Operation * def{argumentInfo.getDefiningOp()}) {
      builder_.setInsertionPointAfter(def);
    } else {
      builder_.setInsertionPointToStart(argumentInfo.getParentBlock());
    }
  } else if (CollectTypeVariables(argument, variables) && variables.empty()) {
    auto proc{mlir::dyn_cast<merk::ProcOp>(builder_.getBlock()->getParentOp())};
    if (!proc) {
      proc =
          builder_.getBlock()->getParentOp()->getParentOfType<merk::ProcOp>();
    }
    builder_.setInsertionPointToStart(&proc.getBody().front());
    argumentInfo = TypeInfoOfType(argument, loc, {}, true);
    if (!argumentInfo) {
      return false;
    }
  } else {
    // Temporary (open types step 5, docs/INPROGRESS.md): T's type_info
    // would be a cell made on the heap, a safepoint too. Liveness reports
    // the root whose type it does not know.
    return false;
  }
  std::optional<std::string> constructor{TypeConstructorName(*type)};
  if (!constructor || !EnsureDescriptor(*type, {})) {
    return false;
  }
  auto infoType{merk::TypeInfoType::get(&context_)};
  mlir::Value descriptor{merk::TypeInfoOp::create(
      builder_, loc, infoType, *constructor, mlir::ValueRange{})};
  mlir::Value info{merk::ForeignCallOp::create(builder_, loc,
      mlir::TypeRange{infoType}, "merk_open_type_info",
      mlir::ValueRange{descriptor, argumentInfo}, /*can_fail=*/false)
                       .getResult(0)};
  builder_.restoreInsertionPoint(here);
  merk::RootTypeOp::create(builder_, loc, value, info);
  return true;
}

// Open types have no order, which binding would change (design document,
// "Open terms", Operations): a comparison that reaches one through a type
// variable stops the program, as one of closures does.
mlir::Value MerkGenerator::OpenCompare(mlir::Location loc) {
  merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
      "merk_open_compared", mlir::ValueRange{}, /*can_fail=*/false);
  return ComparisonValue("=", loc);
}

} // namespace merkurius::codegen::mlirgen
