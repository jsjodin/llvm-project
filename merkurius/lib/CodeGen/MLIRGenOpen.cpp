//===-- lib/CodeGen/MLIRGenOpen.cpp - Open terms --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Open terms (design document, Language extensions, "Open terms"). A value of
// open(T) is a word of T's open layout, which the runtime's generic walkers
// take with T's type_info (runtime/open.h): full unification of a ground T with
// known constructors calls T's own unify procedure (OpenUnifyProc), and of any
// other T merk_open_unify, a fresh variable is merk_open_new_var's cell,
// and a goal that may bind an older variable marks the trail where it can
// fail (MLIRGenControl.cpp). Constructions, deconstructions and get_structure
// are inline, on cells of open.open(T), a type_def of T's constructors with
// every field a word, which lowering lays out as T's open layout.
//
//===----------------------------------------------------------------------===//

#include "MLIRGenImpl.h"

namespace merkurius::codegen::mlirgen {

namespace {
// A variable's primary tag in every open layout, and the primary tag's
// bits, past which a Shifted value is (runtime/open.h).
constexpr std::int64_t kOpenVariableTag{7};
constexpr std::int64_t kOpenTagBits{3};
// Past this many levels of fields other than the last, a type's unify
// procedure gives the rest of the term to the runtime's walker, which
// remembers the cells it has been at (OpenUnifyProc).
constexpr std::int64_t kOpenDepthBeforeWalker{1 << 14};

bool HasTypeVariables(const semantics::Type *type) {
  if (!type) {
    return false;
  }
  if (type->kind == semantics::TypeKind::Variable ||
      type->kind == semantics::TypeKind::Existential) {
    return true;
  }
  return llvm::any_of(type->arguments, HasTypeVariables) ||
      HasTypeVariables(type->result);
}
} // namespace

bool MerkGenerator::IsOpenType(const semantics::Type *type) {
  return types_.types().IsOpen(type);
}

bool MerkGenerator::IsClosedType(const semantics::Type *type) {
  return types_.types().IsClosed(type);
}

bool MerkGenerator::MentionsOpenType(const semantics::Type *type) {
  return types_.types().MentionsOpen(type);
}

// Whether a commit in the procedure releases trail marks without
// -fopen-types: one whose variables may be bound. A variable whose inst has
// `any` in it may be, whatever its type says, such as one of a type variable
// that is open(U) at the call. Choice points mark by their goals instead
// (BindsOlder).
bool MerkGenerator::Trails(const ProcedureBody &body) {
  auto [it, inserted]{trails_.try_emplace(&body, false)};
  if (inserted) {
    it->second = llvm::any_of(body.vars, [&](const semantics::BodyVarInfo &v) {
      return v.any || MentionsOpenType(v.type);
    });
  }
  return it->second;
}

bool MerkGenerator::BindsOlder(const Body &goal, const ProcedureBody &body) {
  if (!Trails(body)) {
    return false;
  }
  llvm::SmallVector<BodyVar> named{
      goal.arguments.begin(), goal.arguments.end()};
  if (goal.kind == BodyKind::Unify) {
    named.push_back(goal.left);
    if (goal.isVar) {
      named.push_back(goal.var);
    }
  }
  if (goal.kind == BodyKind::HOCall) {
    named.push_back(goal.closure);
  }
  for (BodyVar var : named) {
    const semantics::BodyVarInfo &info{body.vars[var]};
    if (values_.lookup(var) && (info.any || MentionsOpenType(info.type))) {
      return true;
    }
  }
  if (goal.lambda && BindsOlder(*goal.lambda, body)) {
    return true;
  }
  for (const Body *inner : goal.goals) {
    if (BindsOlder(*inner, body)) {
      return true;
    }
  }
  for (const semantics::SwitchCase &c : goal.cases) {
    if (BindsOlder(*c.goal, body)) {
      return true;
    }
  }
  return false;
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
      closedWords_.insert(var);
    }
    values_[var] = merk::ForeignCallOp::create(builder_, loc,
        mlir::TypeRange{merk::WordType::get(&context_)}, "merk_open_new_var",
        mlir::ValueRange{}, /*can_fail=*/false)
                       .getResult(0);
  }
}

bool MerkGenerator::IsOpenUnify(const Body &goal, const ProcedureBody &body) {
  if (goal.unify == UnifyKind::GetStructure || goal.unify == UnifyKind::Unify) {
    return true;
  }
  if (goal.unify == UnifyKind::Call || goal.unify == UnifyKind::Assign) {
    return false;
  }
  if (!goal.isVar && IsOpenType(body.vars[goal.left].type)) {
    return true;
  }
  // A deconstruction or test of a closed(T) that holds an open word; its
  // construction is T's, which NoteClosedValues boxes.
  if (!IsClosedType(body.vars[goal.left].type) ||
      goal.unify == UnifyKind::Construct) {
    return false;
  }
  return mlir::isa_and_nonnull<merk::WordType>(
             ValueOf(goal.left) ? ValueOf(goal.left).getType() : nullptr) ||
      (goal.isVar && ValueOf(goal.var) &&
          mlir::isa<merk::WordType>(ValueOf(goal.var).getType()));
}

bool MerkGenerator::ClosedOpen(const ProcedureBody &body, BodyVar var) {
  return IsClosedType(body.vars[var].type) &&
      (body.vars[var].any || closedWords_.contains(var));
}

bool MerkGenerator::OpenParameter(
    const semantics::ArgMode &mode, const semantics::Type *type) {
  return IsClosedType(type) &&
      ((mode.initial && modes_.ops().HasAny(mode.initial, type)) ||
          (mode.final && modes_.ops().HasAny(mode.final, type)));
}

mlir::Value MerkGenerator::ClosedAs(mlir::Value value, bool open,
    const semantics::Type *closed, mlir::Location loc, parser::CharBlock at) {
  if (!value || open == mlir::isa<merk::WordType>(value.getType())) {
    return value;
  }
  if (open) {
    return ToOpenWord(value, closed, loc, at);
  }
  auto word{merk::WordType::get(&context_)};
  mlir::Value ground{
      merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{word},
          "merk_open_closed_value", mlir::ValueRange{value}, /*can_fail=*/false)
          .getResult(0)};
  return Coerce(ground, Represent(closed)->type, loc);
}

void MerkGenerator::NoteClosedValues(
    const Body &goal, const ProcedureBody &body) {
  mlir::Location loc{Location(goal.tokens)};
  parser::CharBlock at{Where(goal.tokens)};
  llvm::SmallVector<BodyVar> vars{goal.arguments.begin(), goal.arguments.end()};
  if (goal.kind == BodyKind::Unify) {
    vars.push_back(goal.left);
    if (goal.isVar) {
      vars.push_back(goal.var);
    }
  }
  for (BodyVar var : vars) {
    mlir::Value value{ValueOf(var)};
    if (!value || !IsClosedType(body.vars[var].type)) {
      continue;
    }
    if (mlir::isa<merk::WordType>(value.getType())) {
      closedWords_.insert(var);
    } else if (ClosedOpen(body, var)) {
      values_[var] = ClosedAs(value, true, body.vars[var].type, loc, at);
    }
  }
}

// X with closed(T)'s open word: X = closed(Y) is a deconstruction or a test
// of the box, or a get_structure, which binds an unbound X to a box of Y's
// word; X = Y is merk_open_unify with closed(T)'s type_info.
void MerkGenerator::ClosedUnify(const Body &goal, const ProcedureBody &body) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  const semantics::Type *type{body.vars[goal.left].type};
  const semantics::Type *argument{type->arguments.front()};
  auto word{merk::WordType::get(&context_)};
  auto i64{builder_.getI64Type()};
  if (goal.isVar) {
    mlir::Value info{TypeInfoOfType(type, loc, at)};
    mlir::Value x{ClosedAs(ValueOf(goal.left), true, type, loc, at)};
    mlir::Value y{ClosedAs(ValueOf(goal.var), true, type, loc, at)};
    if (!info || !x || !y) {
      failed_ = true;
      return;
    }
    auto call{merk::ForeignCallOp::create(builder_, loc,
        mlir::TypeRange{builder_.getI1Type()}, "merk_open_unify",
        mlir::ValueRange{info, x, y}, /*can_fail=*/true)};
    merk::GuardOp::create(builder_, loc, call.getResult(0));
    return;
  }
  BodyVar inner{goal.arguments.front()};
  mlir::Value given{ValueOf(inner)};
  std::optional<Represented> represented{Represent(argument)};
  if (!represented) {
    failed_ = true;
    return;
  }
  if (goal.unify != UnifyKind::GetStructure) {
    // X is bound: its box's word is Y's, or is tested against it.
    mlir::Value value{Coerce(ClosedAs(ValueOf(goal.left), false, type, loc, at),
        represented->type, loc)};
    if (given) {
      Test(goal, value, given, argument);
    } else {
      values_[inner] = value;
    }
    return;
  }
  mlir::Value deref{OpenDeref(ValueOf(goal.left), type, loc)};
  if (!deref) {
    return;
  }
  auto tag{mlir::arith::AndIOp::create(builder_, loc, Coerce(deref, i64, loc),
      mlir::arith::ConstantIntOp::create(
          builder_, loc, i64, kOpenVariableTag))};
  mlir::Value unbound{mlir::arith::CmpIOp::create(builder_, loc,
      mlir::arith::CmpIPredicate::eq, tag,
      mlir::arith::ConstantIntOp::create(
          builder_, loc, i64, kOpenVariableTag))};
  llvm::SmallVector<mlir::Type> results;
  if (!given) {
    results.push_back(represented->type);
  }
  auto ite{merk::IteOp::create(builder_, loc, results)};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
  merk::GuardOp::create(builder_, loc, unbound);
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
  if (!given) {
    // An unbound X has no ground value to give Y (design document, "Open
    // terms", Operations).
    merk::FailOp::create(builder_, loc);
  } else {
    mlir::Value box{
        ToOpenWord(Coerce(given, Represent(type)->type, loc), type, loc, at)};
    if (!box) {
      return;
    }
    merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
        "merk_open_bind", mlir::ValueRange{deref, box}, /*can_fail=*/false);
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  }
  builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
  mlir::Value value{Coerce(
      merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{word},
          "merk_open_closed_value", mlir::ValueRange{deref}, /*can_fail=*/false)
          .getResult(0),
      represented->type, loc)};
  if (given) {
    Test(goal, value, given, argument);
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  } else {
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{value});
  }
  builder_.setInsertionPointAfter(ite);
  if (!given) {
    values_[inner] = ite.getResult(0);
  }
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

// Deref, bind, new variables and boxes are runtime calls here, which
// lowering inlines on the host (Lowering.cpp, LowerOpenCall).
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
  // A closed(T) that may be unbound holds its open word already (ClosedOpen).
  if (IsOpenType(type) ||
      (IsClosedType(type) && mlir::isa<merk::WordType>(value.getType()))) {
    return Coerce(value, word, loc);
  }
  if (std::optional<ScalarEncoding> scalar{OpenScalar(type)}) {
    auto i64{builder_.getI64Type()};
    mlir::Value open;
    if (scalar->kind == ScalarEncoding::Kind::Shifted) {
      // Its bits, zero-extended, past the primary tag.
      mlir::Value bits{Coerce(Coerce(value, word, loc), i64, loc)};
      if (scalar->bits != 0) {
        bits = mlir::arith::AndIOp::create(builder_, loc, bits,
            mlir::arith::ConstantIntOp::create(builder_, loc, i64,
                static_cast<std::int64_t>(
                    (std::uint64_t{1} << scalar->bits) - 1)));
      }
      open = Coerce(mlir::arith::ShLIOp::create(builder_, loc, bits,
                        mlir::arith::ConstantIntOp::create(
                            builder_, loc, i64, kOpenTagBits)),
          word, loc);
    } else {
      open = merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{word},
          "merk_open_box", mlir::ValueRange{Coerce(value, word, loc)},
          /*can_fail=*/false)
                 .getResult(0);
    }
    return NotedOpenWord(open, types_.types().Open(type));
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

std::optional<MerkGenerator::ScalarEncoding> MerkGenerator::OpenScalar(
    const semantics::Type *type) {
  using Kind = ScalarEncoding::Kind;
  if (IsClosedType(type)) {
    return ScalarEncoding{Kind::ClosedBox};
  }
  if (type->kind != semantics::TypeKind::Named) {
    return std::nullopt;
  }
  llvm::StringRef name{type->constructor->name};
  if (type->constructor->module.empty()) {
    if (name == "int" || name == "uint" || name == "int64" ||
        name == "uint64" || name == "float" || name == "string") {
      return ScalarEncoding{Kind::Boxed};
    }
    if (name == "int8" || name == "uint8") {
      return ScalarEncoding{Kind::Shifted, 8};
    }
    if (name == "int16" || name == "uint16") {
      return ScalarEncoding{Kind::Shifted, 16};
    }
    if (name == "int32" || name == "uint32" || name == "character") {
      return ScalarEncoding{Kind::Shifted, 32};
    }
    return std::nullopt;
  }
  // An enumeration, whose ordinary word is its constant's number; one of a
  // single constant is a dummy, 0 in both layouts, as a shift of 0 is.
  const std::vector<semantics::ConstructorTypes> *constructors{
      modes_.ops().ConstructorsOf(type)};
  if (constructors && !constructors->empty() &&
      llvm::all_of(*constructors, [](const semantics::ConstructorTypes &c) {
        return c.arguments.empty();
      })) {
    return ScalarEncoding{Kind::Shifted, 0};
  }
  return std::nullopt;
}

mlir::Value MerkGenerator::FromOpenWord(mlir::Value value,
    const semantics::Type *type, mlir::Location loc, parser::CharBlock at) {
  auto word{merk::WordType::get(&context_)};
  std::optional<Represented> represented{Represent(type)};
  if (!represented) {
    failed_ = true;
    return nullptr;
  }
  mlir::Value result;
  if (std::optional<ScalarEncoding> scalar{OpenScalar(type)}) {
    // Fails if it is an unbound variable, tag 7, which no value has.
    auto i64{builder_.getI64Type()};
    mlir::Value deref{OpenDeref(value, types_.types().Open(type), loc)};
    if (!deref) {
      return nullptr;
    }
    mlir::Value bits{Coerce(deref, i64, loc)};
    mlir::Value tag{mlir::arith::AndIOp::create(builder_, loc, bits,
        mlir::arith::ConstantIntOp::create(
            builder_, loc, i64, kOpenVariableTag))};
    merk::GuardOp::create(builder_, loc,
        mlir::arith::CmpIOp::create(builder_, loc,
            mlir::arith::CmpIPredicate::ne, tag,
            mlir::arith::ConstantIntOp::create(
                builder_, loc, i64, kOpenVariableTag)));
    mlir::Value ordinary{scalar->kind == ScalarEncoding::Kind::Shifted
            ? Coerce(mlir::arith::ShRUIOp::create(builder_, loc, bits,
                         mlir::arith::ConstantIntOp::create(
                             builder_, loc, i64, kOpenTagBits)),
                  word, loc)
            : merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{word},
                  "merk_open_unbox", mlir::ValueRange{deref},
                  /*can_fail=*/false)
                  .getResult(0)};
    result = Coerce(ordinary, represented->type, loc);
  } else {
    mlir::Value info{TypeInfoOfType(type, loc, at)};
    if (!info) {
      failed_ = true;
      return nullptr;
    }
    auto call{merk::ForeignCallOp::create(builder_, loc,
        mlir::TypeRange{builder_.getI1Type(), word}, "merk_open_from_open",
        mlir::ValueRange{info, Coerce(value, word, loc)}, /*can_fail=*/true)};
    merk::GuardOp::create(builder_, loc, call.getResult(0));
    result = Coerce(call.getResult(1), represented->type, loc);
  }
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
      value = IsOpenType(type) || IsClosedType(type)
          ? field
          : FromOpenWord(field, type, loc, at);
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
      bool closed{IsClosedType(type)};
      types.push_back(closed ? mlir::Type{merk::WordType::get(&context_)}
                             : represented->type);
      groundFree |= !IsOpenType(type) && !closed;
    }
  }
  auto tag{mlir::arith::AndIOp::create(builder_, loc, Coerce(deref, i64, loc),
      mlir::arith::ConstantIntOp::create(
          builder_, loc, i64, kOpenVariableTag))};
  mlir::Value unbound{mlir::arith::CmpIOp::create(builder_, loc,
      mlir::arith::CmpIPredicate::eq, tag,
      mlir::arith::ConstantIntOp::create(
          builder_, loc, i64, kOpenVariableTag))};
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
  if (IsClosedType(type)) {
    ClosedUnify(goal, body);
    return;
  }
  if (!goal.isVar) {
    unsigned arity{static_cast<unsigned>(goal.arguments.size())};
    std::optional<merk::TermType> term{OpenTermType(type)};
    if (!term ||
        !ConstructorOf(type->arguments.front(), goal.functor.name, arity)) {
      NotYet(at, "this unification of an open term");
      return;
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
  // Full unification. An ordinary type holds a closed(U) part as U's word,
  // never unbound (ClosedAs), so its own unify entry does, even with open
  // parts.
  if (ReportOpenShapes(goal, {goal.left, goal.var}, body, true)) {
    return;
  }
  Test(goal, ValueOf(goal.left), ValueOf(goal.var), type);
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
      // Temporary (docs/INPROGRESS.md, Open types: what is left): the runtime's
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
  // These read no message, or copy it as copy_term does (open.send).
  llvm::StringRef name{callee.symbol->name};
  if (name == "new_mailbox" || name == "send" || name == "send_any" ||
      name == "close_mailbox" || name.starts_with("spawn")) {
    return false;
  }
  for (BodyVar var : arguments) {
    const semantics::Type *type{body.vars[var].type};
    if (MentionsOpenType(type)) {
      // A receive gives a ground message, which one sent by open.send is
      // not.
      NotYet(Where(goal.tokens),
          "receiving messages with open terms in them but by open.receive, "
          "such as values of type " +
              semantics::TypeToString(*type));
      return true;
    }
  }
  return false;
}

void MerkGenerator::OpenEqual(mlir::Value x, mlir::Value y,
    const semantics::Type *semantic, mlir::Location loc) {
  auto word{merk::WordType::get(&context_)};
  OpenUnifyWords(
      Coerce(x, word, loc), Coerce(y, word, loc), semantic, nullptr, loc);
}

bool MerkGenerator::OpenUnifyWords(mlir::Value x, mlir::Value y,
    const semantics::Type *type, const OpenWalk *walk, mlir::Location loc) {
  auto word{merk::WordType::get(&context_)};
  auto i64{builder_.getI64Type()};
  type = OpenUnifiedAs(type);
  std::optional<ScalarEncoding> scalar{OpenScalar(type)};
  if (scalar && scalar->kind != ScalarEncoding::Kind::ClosedBox) {
    const semantics::Type *open{types_.types().Open(type)};
    mlir::Value a{OpenDeref(x, open, loc)};
    mlir::Value b{OpenDeref(y, open, loc)};
    mlir::Operation *outer{OpenUnifyUnlessBound(a, b, loc)};
    if (scalar->kind == ScalarEncoding::Kind::Shifted) {
      merk::FailOp::create(builder_, loc); // unequal words
    } else {
      // Boxes of equal values: a float's as == has it, a string's by its
      // characters.
      mlir::Type represented{Represent(type)->type};
      llvm::SmallVector<mlir::Value, 2> values;
      for (mlir::Value box : {a, b}) {
        mlir::Value value{Coerce(
            merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{word},
                "merk_open_unbox", mlir::ValueRange{box}, /*can_fail=*/false)
                .getResult(0),
            represented, loc)};
        if (!target_.options().IsDevice() && NoteRoot(value, type)) {
          annotated_.insert(value);
        }
        values.push_back(value);
      }
      Equal(values[0], values[1], type, loc);
      merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    }
    builder_.setInsertionPointAfter(outer);
    return !failed_;
  }
  if (!scalar) {
    if (merk::ProcOp proc{OpenUnifyProc(type)}) {
      OpenWalk start;
      if (!walk) {
        mlir::Value zero{IntegerConstant(i64, 0, loc)};
        start = {zero, zero, zero, IntegerConstant(i64, 1, loc), zero};
        walk = &start;
      }
      merk::CallOp::create(builder_, loc, mlir::TypeRange{}, proc.getSymName(),
          mlir::ValueRange{x, y, walk->savedX, walk->savedY, walk->steps,
              walk->limit, walk->depth});
      return true;
    }
  }
  // A closed(U)'s box holds a ground U, which the walker compares; the
  // occurs check, unify_oc's, is the walker's too.
  mlir::Value info{TypeInfoOfType(type, loc, equalAt_)};
  if (!info) {
    failed_ = true;
    return false;
  }
  auto call{merk::ForeignCallOp::create(builder_, loc,
      mlir::TypeRange{builder_.getI1Type()}, "merk_open_unify",
      mlir::ValueRange{info, x, y}, /*can_fail=*/true)};
  merk::GuardOp::create(builder_, loc, call.getResult(0));
  return true;
}

// A type whose open word is another's, as the runtime's encoding
// AsArgument has it: open(U)'s is U's, and a notag's, with one constructor
// of one argument, is its argument's. A type's unify procedure is never
// one of these, so a pair of words is not met again at once along a walk.
const semantics::Type *MerkGenerator::OpenUnifiedAs(
    const semantics::Type *type) {
  for (;;) {
    if (IsOpenType(type)) {
      type = type->arguments.front();
      continue;
    }
    if (IsClosedType(type) || HasTypeVariables(type)) {
      return type;
    }
    const std::vector<semantics::ConstructorTypes> *constructors{
        modes_.ops().ConstructorsOf(type)};
    if (!constructors || constructors->size() != 1 ||
        constructors->front().arguments.size() != 1) {
      return type;
    }
    llvm::SmallVector<const semantics::Type *> arguments{
        ArgumentsOf(type, constructors->front())};
    if (HasTypeVariables(arguments.front())) {
      return type; // existentially typed: the walker's
    }
    type = arguments.front();
  }
}

// The procedure's arguments are the two words and the walk's state
// (OpenWalk). A pair of cells met again in the walk along last fields is
// one being unified already, whose other fields have been, as the walker's
// remembered pairs are: Brent's cycle detection finds it, with the pair
// saved at each power of two of steps. Other fields are unified one level
// deeper, each with a walk of its own, and past kOpenDepthBeforeWalker
// levels the walker takes the rest, so that a cycle through them ends too.
// The saved pair is two integers, no roots: unification makes no cells, so
// no collection happens in it.
merk::ProcOp MerkGenerator::OpenUnifyProc(const semantics::Type *type) {
  std::string name{"unify_open." + semantics::TypeToString(*type)};
  if (auto it{unifyProcs_.find(name)}; it != unifyProcs_.end()) {
    return it->second;
  }
  unifyProcs_[name] = nullptr;
  if (target_.options().IsDevice() || HasTypeVariables(type)) {
    return nullptr;
  }
  const semantics::Type *open{types_.types().Open(type)};
  std::optional<merk::TermType> term{OpenTermType(open)};
  const std::vector<semantics::ConstructorTypes> *constructors{
      modes_.ops().ConstructorsOf(type)};
  if (!term || !constructors || constructors->empty()) {
    return nullptr;
  }
  llvm::SmallVector<llvm::SmallVector<const semantics::Type *>> arguments;
  for (const semantics::ConstructorTypes &constructor : *constructors) {
    arguments.push_back(ArgumentsOf(type, constructor));
    if (llvm::any_of(arguments.back(), HasTypeVariables)) {
      return nullptr; // an existentially typed constructor
    }
  }
  mlir::Location loc{builder_.getUnknownLoc()};
  auto word{merk::WordType::get(&context_)};
  auto i64{builder_.getI64Type()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto proc{merk::ProcOp::create(builder_, loc, name,
      builder_.getFunctionType({word, word, i64, i64, i64, i64, i64}, {}),
      merk::Determinism::Semidet)};
  proc.setPrivate();
  // Recorded first, for the calls of it on its terms' arguments.
  unifyProcs_[name] = proc;
  mlir::Block *entry{proc.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  auto saved{std::move(typeInfos_)};
  typeInfos_.clear();
  OpenWalk walk{entry->getArgument(2), entry->getArgument(3),
      entry->getArgument(4), entry->getArgument(5), entry->getArgument(6)};
  mlir::Value a{OpenDeref(entry->getArgument(0), open, loc)};
  mlir::Value b{OpenDeref(entry->getArgument(1), open, loc)};
  mlir::Operation *outer{OpenUnifyUnlessBound(a, b, loc)};
  if (OpenUnifyCells(a, b, type, *term, *constructors, arguments, walk, loc)) {
    builder_.setInsertionPointAfter(outer);
    merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  }
  typeInfos_ = std::move(saved);
  return proc;
}

bool MerkGenerator::OpenUnifyCells(mlir::Value a, mlir::Value b,
    const semantics::Type *type, merk::TermType term,
    llvm::ArrayRef<semantics::ConstructorTypes> constructors,
    llvm::ArrayRef<llvm::SmallVector<const semantics::Type *>> arguments,
    const OpenWalk &walk, mlir::Location loc) {
  auto word{merk::WordType::get(&context_)};
  auto i64{builder_.getI64Type()};
  mlir::Value x{Coerce(a, i64, loc)};
  mlir::Value y{Coerce(b, i64, loc)};
  // Too deep: the walker unifies the rest.
  auto deep{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToStart(&deep.getCondition().emplaceBlock());
  merk::GuardOp::create(builder_, loc,
      CompareInts(mlir::arith::CmpIPredicate::sgt, walk.depth,
          IntegerConstant(i64, kOpenDepthBeforeWalker, loc), loc));
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToStart(&deep.getThenRegion().emplaceBlock());
  mlir::Value info{TypeInfoOfType(type, loc, equalAt_)};
  if (!info) {
    failed_ = true;
    return false;
  }
  merk::GuardOp::create(builder_, loc,
      merk::ForeignCallOp::create(builder_, loc,
          mlir::TypeRange{builder_.getI1Type()}, "merk_open_unify",
          mlir::ValueRange{info, a, b}, /*can_fail=*/true)
          .getResult(0));
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToStart(&deep.getElseRegion().emplaceBlock());
  // The saved pair met again: a cycle, unified already.
  auto again{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToStart(&again.getCondition().emplaceBlock());
  merk::GuardOp::create(builder_, loc,
      mlir::arith::AndIOp::create(builder_, loc,
          CompareInts(mlir::arith::CmpIPredicate::eq, x, walk.savedX, loc),
          CompareInts(mlir::arith::CmpIPredicate::eq, y, walk.savedY, loc)));
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToStart(&again.getThenRegion().emplaceBlock());
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToStart(&again.getElseRegion().emplaceBlock());
  // This pair is saved, and the limit doubled, when the steps reach it.
  mlir::Value steps{mlir::arith::AddIOp::create(
      builder_, loc, walk.steps, IntegerConstant(i64, 1, loc))};
  mlir::Value save{
      CompareInts(mlir::arith::CmpIPredicate::eq, steps, walk.limit, loc)};
  mlir::Value zero{IntegerConstant(i64, 0, loc)};
  OpenWalk last{
      mlir::arith::SelectOp::create(builder_, loc, save, x, walk.savedX),
      mlir::arith::SelectOp::create(builder_, loc, save, y, walk.savedY),
      mlir::arith::SelectOp::create(builder_, loc, save, zero, steps),
      mlir::arith::SelectOp::create(builder_, loc, save,
          mlir::arith::ShLIOp::create(
              builder_, loc, walk.limit, IntegerConstant(i64, 1, loc)),
          walk.limit),
      walk.depth};
  OpenWalk deeper{zero, zero, zero, IntegerConstant(i64, 1, loc),
      mlir::arith::AddIOp::create(
          builder_, loc, walk.depth, IntegerConstant(i64, 1, loc))};
  llvm::SmallVector<mlir::Attribute> cases;
  for (const semantics::ConstructorTypes &constructor : constructors) {
    cases.push_back(builder_.getArrayAttr({merk::FunctorAttr::get(
        &context_, constructor.functor.name, constructor.functor.arity)}));
  }
  mlir::Value p{Coerce(a, term, loc)};
  mlir::Value q{Coerce(b, term, loc)};
  auto op{merk::SwitchOp::create(builder_, loc, mlir::TypeRange{}, p,
      builder_.getArrayAttr(cases), /*canFail=*/false)};
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  for (auto [region, attribute, fieldTypes] :
      llvm::zip(op.getCaseRegions(), cases, arguments)) {
    builder_.setInsertionPointToStart(&region.emplaceBlock());
    auto functor{mlir::cast<merk::FunctorAttr>(
        mlir::cast<mlir::ArrayAttr>(attribute)[0])};
    if (constructors.size() > 1) {
      merk::GuardOp::create(
          builder_, loc, merk::HasFunctorOp::create(builder_, loc, q, functor));
    }
    if (!fieldTypes.empty()) {
      llvm::SmallVector<mlir::Type> fields(fieldTypes.size(), word);
      auto ps{merk::DeconstructOp::create(builder_, loc, fields, p, functor)};
      auto qs{merk::DeconstructOp::create(builder_, loc, fields, q, functor)};
      for (auto [i, fieldType] : llvm::enumerate(fieldTypes)) {
        mlir::Value f{ps.getFields()[i]};
        mlir::Value g{qs.getFields()[i]};
        const semantics::Type *fieldOpen{
            IsOpenType(fieldType) ? fieldType : types_.types().Open(fieldType)};
        NotedOpenWord(f, fieldOpen);
        NotedOpenWord(g, fieldOpen);
        bool isLast{i + 1 == fieldTypes.size()};
        if (!OpenUnifyWords(f, g, fieldType, isLast ? &last : &deeper, loc)) {
          return false;
        }
      }
    }
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  }
  return true;
}

mlir::Operation *MerkGenerator::OpenUnifyUnlessBound(
    mlir::Value a, mlir::Value b, mlir::Location loc) {
  auto i64{builder_.getI64Type()};
  // a == b, a unbound and b unbound, in turn, each in the last's else.
  std::array<mlir::Value, 3> tests{
      CompareInts(mlir::arith::CmpIPredicate::eq, Coerce(a, i64, loc),
          Coerce(b, i64, loc), loc),
      IsOpenVariable(a, loc), IsOpenVariable(b, loc)};
  mlir::Operation *outer{nullptr};
  for (unsigned i{0}; i < tests.size(); ++i) {
    auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
    if (!outer) {
      outer = ite;
    } else {
      merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    }
    builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
    merk::GuardOp::create(builder_, loc, tests[i]);
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
    if (i != 0) {
      mlir::Value variable{i == 1 ? a : b};
      mlir::Value value{i == 1 ? b : a};
      merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
          "merk_open_bind", mlir::ValueRange{variable, value},
          /*can_fail=*/false);
    }
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
  }
  return outer;
}

mlir::Value MerkGenerator::IsOpenVariable(
    mlir::Value word, mlir::Location loc) {
  auto i64{builder_.getI64Type()};
  mlir::Value tag{mlir::arith::AndIOp::create(builder_, loc,
      Coerce(word, i64, loc), IntegerConstant(i64, kOpenVariableTag, loc))};
  return CompareInts(mlir::arith::CmpIPredicate::eq, tag,
      IntegerConstant(i64, kOpenVariableTag, loc), loc);
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
  bool ground{false};
  if (argument->kind == semantics::TypeKind::Variable) {
    argumentInfo = VariableTypeInfo(argument->var);
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
    ground = true;
  } else {
    // T has type variables under a constructor, as list(T1) has: its
    // type_info is a kept one, made after its variables' (KeptTypeInfo),
    // which must be in one block.
    mlir::Block *block{nullptr};
    mlir::Operation *latest{nullptr};
    for (unsigned var : variables) {
      mlir::Value info{VariableTypeInfo(var)};
      if (!info) {
        return false;
      }
      mlir::Operation *def{info.getDefiningOp()};
      mlir::Block *in{def ? def->getBlock() : info.getParentBlock()};
      if (block && in != block) {
        return false;
      }
      block = in;
      if (def && (!latest || latest->isBeforeInBlock(def))) {
        latest = def;
      }
    }
    if (latest) {
      builder_.setInsertionPointAfter(latest);
    } else {
      builder_.setInsertionPointToStart(block);
    }
    argumentInfo = KeptTypeInfo(argument, loc);
    if (!argumentInfo) {
      return false;
    }
  }
  std::optional<std::string> constructor{TypeConstructorName(*type)};
  if (!constructor || !EnsureDescriptor(*type, {})) {
    return false;
  }
  auto infoType{merk::TypeInfoType::get(&context_)};
  mlir::Value info;
  if (ground) {
    // A ground T's is static, as T's is, so it allocates nothing.
    info = merk::TypeInfoOp::create(
        builder_, loc, infoType, *constructor, mlir::ValueRange{argumentInfo});
  } else {
    // A T with type variables is known only at run time; a type_info made
    // on the heap would be a safepoint, which a note cannot be.
    mlir::Value descriptor{merk::TypeInfoOp::create(
        builder_, loc, infoType, *constructor, mlir::ValueRange{})};
    info = merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{infoType},
        "merk_open_type_info", mlir::ValueRange{descriptor, argumentInfo},
        /*can_fail=*/false)
               .getResult(0);
  }
  openTypeInfos_[{proc, argument}] = info;
  builder_.restoreInsertionPoint(here);
  merk::RootTypeOp::create(builder_, loc, value, info);
  return true;
}

// A type variable's type_info, from the procedure's type_infos or those an
// existential constructor gave; null if neither has it.
mlir::Value MerkGenerator::VariableTypeInfo(unsigned var) {
  mlir::Value info{typeInfos_.lookup(var)};
  return info ? info : givenInfos_.lookup(var);
}

// A type_info of a type with type variables that lasts as long as the
// program (merk_kept_type_info_begin): the runtime makes it from its
// arguments' type_infos, static ones for ground arguments, without a
// safepoint, so that it can describe a root. Null for a type it cannot
// name, which leaves the root to Liveness's report.
mlir::Value MerkGenerator::KeptTypeInfo(
    const semantics::Type *type, mlir::Location loc) {
  if (type->kind == semantics::TypeKind::Variable) {
    return VariableTypeInfo(type->var);
  }
  llvm::SmallVector<unsigned> variables;
  if (!CollectTypeVariables(type, variables)) {
    return {};
  }
  if (variables.empty()) {
    return TypeInfoOfType(type, loc, {}, true);
  }
  if (type->kind != semantics::TypeKind::Named &&
      type->kind != semantics::TypeKind::Tuple) {
    return {};
  }
  std::optional<std::string> constructor{TypeConstructorName(*type)};
  if (!constructor || !EnsureDescriptor(*type, {})) {
    return {};
  }
  auto i64{builder_.getI64Type()};
  auto infoType{merk::TypeInfoType::get(&context_)};
  mlir::Value descriptor{merk::TypeInfoOp::create(
      builder_, loc, infoType, *constructor, mlir::ValueRange{})};
  mlir::Value key{merk::ForeignCallOp::create(builder_, loc,
      mlir::TypeRange{i64}, "merk_kept_type_info_begin",
      mlir::ValueRange{descriptor}, /*can_fail=*/false)
                      .getResult(0)};
  for (const semantics::Type *argument : type->arguments) {
    mlir::Value info{KeptTypeInfo(argument, loc)};
    if (!info) {
      return {};
    }
    key = merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{i64},
        "merk_kept_type_info_argument", mlir::ValueRange{key, info},
        /*can_fail=*/false)
              .getResult(0);
  }
  return merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{infoType},
      "merk_kept_type_info_end", mlir::ValueRange{key}, /*can_fail=*/false)
      .getResult(0);
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
