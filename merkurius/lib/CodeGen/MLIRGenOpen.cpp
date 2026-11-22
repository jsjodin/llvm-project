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
// (MLIRGenControl.cpp). Constructions, deconstructions and get_structure
// are inline, on cells of open.open(T), a type_def of T's constructors with
// every field a word, which lowering lays out as T's open layout.
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

std::optional<merk::TermType> MerkGenerator::OpenTermType(
    const semantics::Type *open) {
  std::string name{semantics::TypeToString(*open)};
  auto term{merk::TermType::get(&context_, name)};
  if (auto it{terms_.find(name)}; it != terms_.end()) {
    return it->second ? std::make_optional(term) : std::nullopt;
  }
  const semantics::Type *type{open->arguments.front()};
  const std::vector<semantics::ConstructorTypes> *constructors{
      modes_.ops().ConstructorsOf(type)};
  if (!constructors || name.front() == '<') {
    terms_[name] = nullptr;
    return std::nullopt;
  }
  terms_[name] = open;
  auto word{merk::WordType::get(&context_)};
  llvm::SmallVector<mlir::Attribute> attributes;
  for (const semantics::ConstructorTypes &constructor : *constructors) {
    llvm::SmallVector<mlir::Type> fields(constructor.arguments.size(), word);
    attributes.push_back(merk::ConstructorAttr::get(
        &context_, constructor.functor.name, fields, 0));
  }
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(module_.getBody());
  for (mlir::Operation &op : *module_.getBody()) {
    if (!mlir::isa<merk::TypeDefOp>(op)) {
      break;
    }
    builder_.setInsertionPointAfter(&op);
  }
  merk::TypeDefOp::create(builder_, builder_.getUnknownLoc(), name,
      builder_.getArrayAttr(attributes), /*abstract=*/false, /*open=*/true);
  return term;
}

// Temporary (open types step 4(d), docs/INPROGRESS.md): deref, bind and
// to_open are runtime calls, rather than inline code for deref and bind and
// a static box for a small int.
mlir::Value MerkGenerator::OpenDeref(
    mlir::Value word, const semantics::Type *open, mlir::Location loc) {
  auto type{merk::WordType::get(&context_)};
  return NotedOpenWord(
      merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{type},
          "merk_open_deref", mlir::ValueRange{Coerce(word, type, loc)},
          /*can_fail=*/false)
          .getResult(0),
      open);
}

mlir::Value MerkGenerator::NotedOpenWord(
    mlir::Value value, const semantics::Type *type) {
  if (!value || failed_ || target_.options().IsDevice()) {
    return value;
  }
  // As AnnotateRoots does, a root it cannot describe is left without a
  // note, which Liveness reports.
  if (!annotated_.contains(value) &&
      NoteOpenRoot(
          value, IsClosedType(type) ? type : type->arguments.front())) {
    annotated_.insert(value);
  }
  return value;
}

mlir::Value MerkGenerator::ToOpenWord(mlir::Value value,
    const semantics::Type *type, mlir::Location loc, parser::CharBlock at) {
  auto word{merk::WordType::get(&context_)};
  if (IsOpenType(type)) {
    return Coerce(value, word, loc);
  }
  mlir::Value info{TypeInfoOfType(type, loc, at)};
  if (!info) {
    failed_ = true;
    return nullptr;
  }
  return NotedOpenWord(
      merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{word},
          "merk_open_to_open", mlir::ValueRange{info, Coerce(value, word, loc)},
          /*can_fail=*/false)
          .getResult(0),
      types_.types().Open(type));
}

mlir::Value MerkGenerator::FromOpenWord(mlir::Value value,
    const semantics::Type *type, mlir::Location loc, parser::CharBlock at) {
  auto word{merk::WordType::get(&context_)};
  std::optional<Represented> represented{Represent(type)};
  mlir::Value info{TypeInfoOfType(type, loc, at)};
  if (!info || !represented) {
    failed_ = true;
    return nullptr;
  }
  auto call{merk::ForeignCallOp::create(builder_, loc,
      mlir::TypeRange{builder_.getI1Type(), word}, "merk_open_from_open",
      mlir::ValueRange{info, Coerce(value, word, loc)}, /*can_fail=*/true)};
  merk::GuardOp::create(builder_, loc, call.getResult(0));
  mlir::Value result{Coerce(call.getResult(1), represented->type, loc)};
  if (!target_.options().IsDevice() && NoteRoot(result, type)) {
    annotated_.insert(result);
  }
  return result;
}

std::optional<llvm::SmallVector<mlir::Value>> MerkGenerator::OpenArguments(
    const Body &goal, const ProcedureBody &body) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  auto word{merk::WordType::get(&context_)};
  llvm::SmallVector<mlir::Value> words;
  for (BodyVar argument : goal.arguments) {
    mlir::Value value{ValueOf(argument)};
    if (!value) {
      // A free open argument of a get_structure's new cell.
      value = NotedOpenWord(
          merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{word},
              "merk_open_new_var", mlir::ValueRange{}, /*can_fail=*/false)
              .getResult(0),
          body.vars[argument].type);
    } else {
      value = ToOpenWord(value, body.vars[argument].type, loc, at);
      if (!value) {
        return std::nullopt;
      }
    }
    words.push_back(value);
  }
  return words;
}

std::optional<llvm::SmallVector<mlir::Value>> MerkGenerator::OpenRead(
    const Body &goal, const ProcedureBody &body, mlir::Value deref,
    merk::TermType term, bool test) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  auto word{merk::WordType::get(&context_)};
  unsigned arity{static_cast<unsigned>(goal.arguments.size())};
  auto functor{merk::FunctorAttr::get(&context_, goal.functor.name, arity)};
  mlir::Value cell{Coerce(deref, term, loc)};
  const std::vector<semantics::ConstructorTypes> *constructors{
      modes_.ops().ConstructorsOf(
          body.vars[goal.left].type->arguments.front())};
  if (test && constructors && constructors->size() > 1) {
    merk::GuardOp::create(builder_, loc,
        merk::HasFunctorOp::create(builder_, loc, cell, functor));
  }
  llvm::SmallVector<mlir::Value> results;
  if (arity == 0) {
    return results;
  }
  auto fields{merk::DeconstructOp::create(builder_, loc,
      llvm::SmallVector<mlir::Type>(arity, word), cell, functor)};
  for (auto [argument, field] : llvm::zip(goal.arguments, fields.getFields())) {
    const semantics::Type *type{body.vars[argument].type};
    const semantics::Type *open{
        IsOpenType(type) ? type : types_.types().Open(type)};
    if (!NotedOpenWord(field, open)) {
      return std::nullopt;
    }
    mlir::Value value{ValueOf(argument)};
    if (!value) {
      value = IsOpenType(type) ? field : FromOpenWord(field, type, loc, at);
      if (!value) {
        return std::nullopt;
      }
      results.push_back(value);
      continue;
    }
    // A bound argument is unified with its field.
    mlir::Value openWord{ToOpenWord(value, type, loc, at)};
    if (!openWord) {
      return std::nullopt;
    }
    OpenEqual(field, openWord, open, loc);
  }
  return results;
}

// X = f(Y1, ..., Yn) with X `any`: if X is an unbound variable, it is bound
// to a new cell, whose free arguments are new variables; a free argument of
// a ground type has no value then, and the unification fails. Otherwise X
// is read as a deconstruction or a test is (design document, "Open
// terms", Operations).
void MerkGenerator::OpenGetStructure(
    const Body &goal, const ProcedureBody &body, merk::TermType term) {
  mlir::Location loc{Location(goal.tokens)};
  auto word{merk::WordType::get(&context_)};
  auto i64{builder_.getI64Type()};
  unsigned arity{static_cast<unsigned>(goal.arguments.size())};
  auto functor{merk::FunctorAttr::get(&context_, goal.functor.name, arity)};
  mlir::Value deref{
      OpenDeref(ValueOf(goal.left), body.vars[goal.left].type, loc)};
  if (!deref) {
    return;
  }
  llvm::SmallVector<BodyVar> free;
  llvm::SmallVector<mlir::Type> types;
  bool groundFree{false};
  for (BodyVar argument : goal.arguments) {
    if (!ValueOf(argument)) {
      const semantics::Type *type{body.vars[argument].type};
      std::optional<Represented> represented{Represent(type)};
      if (!represented) {
        failed_ = true;
        return;
      }
      free.push_back(argument);
      types.push_back(represented->type);
      groundFree |= !IsOpenType(type);
    }
  }
  // A variable's primary tag, in every open layout (runtime/open.h).
  constexpr std::int64_t kVariableTag{7};
  auto tag{mlir::arith::AndIOp::create(builder_, loc, Coerce(deref, i64, loc),
      mlir::arith::ConstantIntOp::create(builder_, loc, i64, kVariableTag))};
  mlir::Value unbound{mlir::arith::CmpIOp::create(builder_, loc,
      mlir::arith::CmpIPredicate::eq, tag,
      mlir::arith::ConstantIntOp::create(builder_, loc, i64, kVariableTag))};
  auto ite{merk::IteOp::create(builder_, loc, types)};
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
    merk::GuardOp::create(builder_, loc, unbound);
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
    if (groundFree) {
      merk::FailOp::create(builder_, loc);
    } else {
      std::optional<llvm::SmallVector<mlir::Value>> words{
          OpenArguments(goal, body)};
      if (!words) {
        return;
      }
      mlir::Value cell{
          merk::ConstructOp::create(builder_, loc, term, functor, *words)};
      merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
          "merk_open_bind", mlir::ValueRange{deref, Coerce(cell, word, loc)},
          /*can_fail=*/false);
      llvm::SmallVector<mlir::Value> made;
      for (auto [argument, value] : llvm::zip(goal.arguments, *words)) {
        if (llvm::is_contained(free, argument)) {
          made.push_back(value);
        }
      }
      merk::YieldOp::create(builder_, loc, made);
    }
    builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
    std::optional<llvm::SmallVector<mlir::Value>> read{
        OpenRead(goal, body, deref, term, /*test=*/true)};
    if (!read) {
      return;
    }
    merk::YieldOp::create(builder_, loc, *read);
  }
  for (auto [argument, value] : llvm::zip(free, ite.getResults())) {
    values_[argument] = value;
  }
}

void MerkGenerator::OpenUnify(const Body &goal, const ProcedureBody &body) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  const semantics::Type *type{body.vars[goal.left].type};
  if (!goal.isVar) {
    if (!IsOpenType(type)) {
      // Temporary (open types step 4(c), docs/INPROGRESS.md).
      NotYet(at, "closed(T) values that may be unbound");
      return;
    }
    unsigned arity{static_cast<unsigned>(goal.arguments.size())};
    std::optional<merk::TermType> term{OpenTermType(type)};
    if (!term ||
        !ConstructorOf(type->arguments.front(), goal.functor.name, arity)) {
      NotYet(at, "this unification of an open term");
      return;
    }
    for (BodyVar argument : goal.arguments) {
      if (!ValueOf(argument) && IsClosedType(body.vars[argument].type)) {
        // Temporary (open types step 4(c), docs/INPROGRESS.md).
        NotYet(at, "closed(T) values that may be unbound");
        return;
      }
    }
    if (ReportOpenShapes(goal, {goal.left}, body, false)) {
      return;
    }
    switch (goal.unify) {
    case UnifyKind::Construct: {
      for (BodyVar argument : goal.arguments) {
        if (!ValueOf(argument)) {
          NotYet(at, "an open term with free arguments");
          return;
        }
      }
      std::optional<llvm::SmallVector<mlir::Value>> words{
          OpenArguments(goal, body)};
      if (!words) {
        return;
      }
      values_[goal.left] = Coerce(
          merk::ConstructOp::create(builder_, loc, *term,
              merk::FunctorAttr::get(&context_, goal.functor.name, arity),
              *words),
          merk::WordType::get(&context_), loc);
      return;
    }
    case UnifyKind::GetStructure:
      OpenGetStructure(goal, body, *term);
      return;
    default: {
      // A deconstruction or a test of a bound X, whose word may still be a
      // bound variable's reference.
      mlir::Value deref{OpenDeref(ValueOf(goal.left), type, loc)};
      if (!deref) {
        return;
      }
      std::optional<llvm::SmallVector<mlir::Value>> read{OpenRead(goal, body,
          deref, *term,
          !goal.switchTest && (goal.canFail || goal.unify == UnifyKind::Test))};
      if (!read) {
        return;
      }
      unsigned i{0};
      for (BodyVar argument : goal.arguments) {
        if (!ValueOf(argument)) {
          values_[argument] = (*read)[i++];
        }
      }
      return;
    }
    }
  }
  // Full unification. A closed(T) that may be unbound is a box or a
  // variable, which its ordinary word is not, outside open(T).
  if (!IsOpenType(type) && MentionsClosedType(type)) {
    // Temporary (open types step 4(c), docs/INPROGRESS.md).
    NotYet(at, "closed(T) values that may be unbound");
    return;
  }
  if (ReportOpenShapes(goal, {goal.left, goal.var}, body, true)) {
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

std::optional<std::string> MerkGenerator::OpenShapeNotYet(
    const semantics::Type *type, bool inOpen, bool closedToo,
    llvm::SmallPtrSetImpl<const semantics::Type *> &seen) {
  if (!type || !seen.insert(type).second) {
    return std::nullopt;
  }
  if (IsOpenType(type)) {
    return OpenShapeNotYet(type->arguments.front(), true, closedToo, seen);
  }
  if (IsClosedType(type)) {
    // A box that to_open and from_open share, and only unification walks.
    if (inOpen && !closedToo) {
      return std::nullopt;
    }
    return OpenShapeNotYet(type->arguments.front(), inOpen, closedToo, seen);
  }
  if (!inOpen) {
    // Outside open(_), only the open parts within count.
    for (const semantics::Type *argument : type->arguments) {
      if (std::optional<std::string> why{
              OpenShapeNotYet(argument, false, closedToo, seen)}) {
        return why;
      }
    }
    return OpenShapeNotYet(type->result, false, closedToo, seen);
  }
  switch (type->kind) {
  case semantics::TypeKind::HigherOrder:
    return "higher-order values";
  case semantics::TypeKind::Variable:
  case semantics::TypeKind::Existential:
    return std::nullopt; // the runtime's check, where it is known
  default:
    break;
  }
  if (type->kind == semantics::TypeKind::Named &&
      type->constructor->module == "array" &&
      type->constructor->name == "array") {
    return "arrays";
  }
  if (type->kind == semantics::TypeKind::Tuple) {
    for (const semantics::Type *element : type->arguments) {
      if (std::optional<std::string> why{
              OpenShapeNotYet(element, true, closedToo, seen)}) {
        return why;
      }
    }
    return std::nullopt;
  }
  const std::vector<semantics::ConstructorTypes> *constructors{
      modes_.ops().ConstructorsOf(type)};
  if (!constructors) {
    return std::nullopt; // a primitive or abstract type, by its descriptor
  }
  for (const semantics::ConstructorTypes &constructor : *constructors) {
    const semantics::Signature *signature{
        constructor.symbol ? types_.SignatureOf(*constructor.symbol) : nullptr};
    if (signature && signature->existentials != 0) {
      return "existentially typed constructors";
    }
    for (const semantics::Type *field : ArgumentsOf(type, constructor)) {
      if (std::optional<std::string> why{
              OpenShapeNotYet(field, true, closedToo, seen)}) {
        return why;
      }
    }
  }
  return std::nullopt;
}

bool MerkGenerator::ReportOpenShapes(const Body &goal,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body,
    bool closedToo) {
  for (BodyVar var : arguments) {
    const semantics::Type *type{body.vars[var].type};
    if (!MentionsOpenType(type)) {
      continue;
    }
    llvm::SmallPtrSet<const semantics::Type *, 8> seen;
    if (std::optional<std::string> why{
            OpenShapeNotYet(type, false, closedToo, seen)}) {
      // Temporary (open types step 4, docs/INPROGRESS.md): the runtime's
      // walkers stop on these; the compiler reports them where it can.
      NotYet(Where(goal.tokens),
          "open terms that hold " + *why + ", as values of type " +
              semantics::TypeToString(*type) + " can");
      return true;
    }
  }
  return false;
}

bool MerkGenerator::ReportOpenMessages(const semantics::Procedure &callee,
    const Body &goal, llvm::ArrayRef<BodyVar> arguments,
    const ProcedureBody &body) {
  if (callee.symbol->module != "process") {
    return false;
  }
  for (BodyVar var : arguments) {
    const semantics::Type *type{body.vars[var].type};
    if (MentionsOpenType(type)) {
      // Temporary (open types step 6, docs/INPROGRESS.md): a message is
      // copied as copy_term does then.
      NotYet(Where(goal.tokens),
          "messages with open terms in them, such as values of type " +
              semantics::TypeToString(*type));
      return true;
    }
  }
  return false;
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
    mlir::Value value, const semantics::Type *argument) {
  mlir::Location loc{value.getLoc()};
  auto proc{mlir::dyn_cast<merk::ProcOp>(builder_.getBlock()->getParentOp())};
  if (!proc) {
    proc = builder_.getBlock()->getParentOp()->getParentOfType<merk::ProcOp>();
  }
  if (mlir::Value info{openTypeInfos_.lookup({proc, argument})}) {
    merk::RootTypeOp::create(builder_, loc, value, info);
    return true;
  }
  // Named by open(T0): open(closed(U)) is closed(U) as a type.
  const semantics::Type *type{types_.types().Open(types_.types().Variable(0))};
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
  openTypeInfos_[{proc, argument}] = info;
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
