//===-- lib/CodeGen/MLIRGenTypeInfo.cpp - Type information ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type descriptors, type_infos and typeclass_infos, instance tables and
// method dispatch.
//
//===----------------------------------------------------------------------===//

#include "MLIRGenImpl.h"

namespace merkurius::codegen::mlirgen {

// Temporary (chunk 1 step 6, docs/INPROGRESS.md): the library's modules not
// compiled yet, whose types' descriptors each module using them makes.
static constexpr llvm::StringLiteral kUncompiledLibrary[]{"array", "bool",
    "builtin", "char", "exception", "int", "int16", "int32", "int64", "int8",
    "string", "uint", "uint16", "uint32", "uint64", "uint8", "univ"};

// A type's arguments in its type_info: a higher-order type's are its
// arguments' types and then a function's result's, as polymorphism analysis
// has them.
static llvm::SmallVector<const semantics::Type *> TypeParts(
    const semantics::Type &type) {
  llvm::SmallVector<const semantics::Type *> parts{
      type.arguments.begin(), type.arguments.end()};
  if (type.result) {
    parts.push_back(type.result);
  }
  return parts;
}

// Whether a type's type_info has the number of its arguments in word 1, as a
// tuple's and a higher-order type's, whose descriptors are for any number.
static bool HasVariableArity(const semantics::Type &type) {
  return type.kind == semantics::TypeKind::Tuple ||
      type.kind == semantics::TypeKind::HigherOrder;
}

// A type constructor's name in merk.type_info, with its arity.
static std::optional<std::string> TypeConstructorName(
    const semantics::Type &type) {
  std::string arity{"/" + std::to_string(TypeParts(type).size())};
  if (type.kind == semantics::TypeKind::Tuple) {
    return "{}" + arity;
  }
  if (type.kind == semantics::TypeKind::HigherOrder) {
    return (type.predOrFunc == parser::PredOrFunc::Function ? "func" : "pred") +
        arity;
  }
  if (type.kind != semantics::TypeKind::Named) {
    return std::nullopt;
  }
  const semantics::Symbol &symbol{*type.constructor};
  std::string name{symbol.name};
  if (!symbol.module.empty()) {
    name = std::string{symbol.module} + "." + name;
  }
  return name + arity;
}

void MerkGenerator::EmitDescriptor(llvm::StringRef constructor,
    llvm::StringRef module, llvm::StringRef name, llvm::StringRef kind,
    llvm::StringRef linkage, unsigned arity, mlir::Type term,
    merk::ProcOp unify, merk::ProcOp compare, mlir::ArrayAttr functors) {
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  merk::TypeCtorOp::create(builder_, builder_.getUnknownLoc(), constructor,
      module, name, kind, linkage, arity,
      term ? mlir::TypeAttr::get(term) : mlir::TypeAttr{},
      unify ? mlir::FlatSymbolRefAttr::get(unify.getSymNameAttr())
            : mlir::FlatSymbolRefAttr{},
      compare ? mlir::FlatSymbolRefAttr::get(compare.getSymNameAttr())
              : mlir::FlatSymbolRefAttr{},
      functors);
}

bool MerkGenerator::EnsureVoidDescriptor() {
  if (descriptors_.insert("void/0").second) {
    // There are no values of it: only a value nothing determines has it, and
    // nothing can compare those, so its entry is never reached.
    EmitDescriptor("void/0", "", "void", "void", "linkonce", 0, {},
        DescriptorEntry(false, "void/0", nullptr, {}, /*dummy=*/true, {}),
        DescriptorEntry(true, "void/0", nullptr, {}, /*dummy=*/true, {}), {});
  }
  return true;
}

merk::ProcOp MerkGenerator::HigherOrderEntry(
    bool compare, llvm::StringRef constructor) {
  mlir::Location loc{builder_.getUnknownLoc()};
  auto info{merk::TypeInfoType::get(&context_)};
  auto word{merk::WordType::get(&context_)};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto proc{merk::ProcOp::create(builder_, loc,
      ((compare ? "compare_entry." : "unify_entry.") + constructor).str(),
      compare ? builder_.getFunctionType(
                    {info, word, word}, mlir::Type{*ComparisonType()})
              : builder_.getFunctionType({info, word, word}, {}),
      compare ? merk::Determinism::Det : merk::Determinism::Semidet)};
  proc.setPrivate();
  mlir::Block *entry{proc.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  // It does not return.
  merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
      "merk_higher_order_called",
      mlir::ValueRange{IntegerConstant(builder_.getI32Type(), compare, loc)},
      /*can_fail=*/false);
  if (compare) {
    merk::SucceedOp::create(builder_, loc, ComparisonValue("=", loc));
  } else {
    merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  }
  return proc;
}

// The entry a generic unification or comparison calls: the type's arguments'
// type_infos out of its own, and its values out of their words.
merk::ProcOp MerkGenerator::DescriptorEntry(bool compare,
    llvm::StringRef constructor,
    const semantics::Type *canonical, mlir::Type represented, bool dummy,
    parser::CharBlock at) {
  mlir::Location loc{builder_.getUnknownLoc()};
  auto info{merk::TypeInfoType::get(&context_)};
  auto word{merk::WordType::get(&context_)};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto proc{merk::ProcOp::create(builder_, loc,
      ((compare ? "compare_entry." : "unify_entry.") + constructor).str(),
      compare ? builder_.getFunctionType(
                    {info, word, word}, mlir::Type{*ComparisonType()})
              : builder_.getFunctionType({info, word, word}, {}),
      compare ? merk::Determinism::Det : merk::Determinism::Semidet)};
  proc.setPrivate();
  mlir::Block *entry{proc.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  mlir::Value result;
  if (!dummy) {
    auto saved{std::move(typeInfos_)};
    parser::CharBlock savedAt{equalAt_};
    typeInfos_.clear();
    equalAt_ = at;
    // A tuple's cell has the number of its arguments in word 1.
    unsigned first{canonical->kind == semantics::TypeKind::Tuple ? 2u : 1u};
    for (auto [i, argument] : llvm::enumerate(canonical->arguments)) {
      typeInfos_[argument->var] = merk::TypeInfoWordOp::create(builder_, loc,
          info, entry->getArgument(0), first + i);
    }
    mlir::Value x{Coerce(entry->getArgument(1), represented, loc)};
    mlir::Value y{Coerce(entry->getArgument(2), represented, loc)};
    if (compare) {
      result = Compare(x, y, canonical, /*generic=*/true, loc);
    } else {
      Equal(x, y, canonical, loc);
    }
    typeInfos_ = std::move(saved);
    equalAt_ = savedAt;
  } else if (compare) {
    result = ComparisonValue("=", loc);
  }
  merk::SucceedOp::create(builder_, loc,
      result ? mlir::ValueRange{result} : mlir::ValueRange{});
  return proc;
}

bool MerkGenerator::EnsureDescriptor(
    const semantics::Type &type, parser::CharBlock at) {
  std::optional<std::string> constructor{TypeConstructorName(type)};
  if (!constructor) {
    return false;
  }
  if (descriptors_.contains(*constructor)) {
    return true;
  }
  if (type.kind == semantics::TypeKind::HigherOrder) {
    // As mmc, a program that unifies or compares closures stops.
    descriptors_.insert(*constructor);
    EmitDescriptor(*constructor, "",
        type.predOrFunc == parser::PredOrFunc::Function ? "func" : "pred",
        "closure", "linkonce", TypeParts(type).size(), {},
        HigherOrderEntry(false, *constructor),
        HigherOrderEntry(true, *constructor), {});
    return true;
  }
  // Its most general instance, which its one unify entry works on.
  semantics::TypeContext &context{types_.types()};
  llvm::SmallVector<const semantics::Type *> variables;
  for (unsigned i{0}; i < type.arguments.size(); ++i) {
    variables.push_back(context.Variable(i));
  }
  const semantics::Type *canonical{type.kind == semantics::TypeKind::Tuple
          ? context.Tuple(variables)
          : context.Named(*type.constructor, variables)};
  std::string module, name{"{}"};
  std::string kind{"term"}, linkage{"linkonce"};
  if (type.kind == semantics::TypeKind::Tuple) {
    kind = "tuple";
  } else {
    module = type.constructor->module;
    name = type.constructor->name;
    if (module.empty()) {
      kind = name == "character" ? "char" : name;
    } else if (module == "array" && name == "array") {
      kind = "array";
    } else if (module == moduleName_) {
      linkage = type.constructor->exported ? "external" : "internal";
      // The state of the world, an abstract type that io does not define, is
      // a dummy whose linkonce descriptor each module using it makes, as
      // below.
    } else if (!llvm::is_contained(kUncompiledLibrary, llvm::StringRef{module}) &&
        !(module == "io" && name == "state")) {
      // The module that defines it has it; this one may not even see its
      // constructors.
      descriptors_.insert(*constructor);
      EmitDescriptor(*constructor, module, name, kind, "declaration",
          type.arguments.size(), {}, {}, {}, {});
      return true;
    }
  }
  std::optional<Represented> represented{Represent(canonical)};
  if (!represented) {
    NotYet(at, "type_infos of type " + semantics::TypeToString(type));
    return false;
  }
  if (represented->dummy) {
    kind = "dummy";
    linkage = "linkonce";
  }
  // Before its entry, whose unification of a recursive type needs it.
  descriptors_.insert(*constructor);
  llvm::StringSet<> seen;
  if (!represented->dummy && !Comparable(represented->type, seen)) {
    NotYet(at, "type_infos of type " + semantics::TypeToString(type));
    return false;
  }
  // A primitive type, or an array, has no constructors, nor has the state
  // of the world, an abstract dummy.
  mlir::ArrayAttr functors;
  if (represented->dummy && !modes_.ops().ConstructorsOf(canonical)) {
    functors = builder_.getArrayAttr({});
  } else if (represented->dummy ||
      mlir::isa_and_nonnull<merk::TermType>(represented->type)) {
    functors = FunctorsAttr(canonical, at);
    if (!functors) {
      failed_ = true;
      return false;
    }
  }
  merk::ProcOp unify{DescriptorEntry(false, *constructor, canonical,
      represented->type, represented->dummy, at)};
  merk::ProcOp compare{DescriptorEntry(true, *constructor, canonical,
      represented->type, represented->dummy, at)};
  EmitDescriptor(*constructor, module, name, kind, linkage,
      type.arguments.size(),
      mlir::isa_and_nonnull<merk::TermType>(represented->type)
          ? represented->type
          : mlir::Type{},
      unify, compare, functors);
  return true;
}

// A type parameter by its number, an existentially quantified type
// variable by the place of its type_info in `hidden`, and any other type as an
// array of its constructor and its arguments, in which their descriptors are
// ensured.
mlir::Attribute MerkGenerator::FieldTypeAttr(const semantics::Type &type,
    parser::CharBlock at, const HiddenFields *hidden) {
  if (type.kind == semantics::TypeKind::Variable) {
    // A canonical type's variables are its parameters, numbered from 0.
    if (hidden && type.var >= kCellExistential) {
      auto [field, word]{hidden->places[type.var - kCellExistential]};
      return builder_.getDenseI64ArrayAttr({field, word});
    }
    return builder_.getI64IntegerAttr(type.var);
  }
  std::optional<std::string> constructor{TypeConstructorName(type)};
  if (!constructor || !EnsureDescriptor(type, at)) {
    if (!constructor) {
      NotYet(at, "type_infos of type " + semantics::TypeToString(type));
    }
    return nullptr;
  }
  llvm::SmallVector<mlir::Attribute> parts{builder_.getStringAttr(*constructor)};
  for (const semantics::Type *argument : TypeParts(type)) {
    mlir::Attribute attribute{FieldTypeAttr(*argument, at, hidden)};
    if (!attribute) {
      return nullptr;
    }
    parts.push_back(attribute);
  }
  return builder_.getArrayAttr(parts);
}

mlir::ArrayAttr MerkGenerator::FunctorsAttr(
    const semantics::Type *canonical, parser::CharBlock at) {
  const std::vector<semantics::ConstructorTypes> *constructors{
      modes_.ops().ConstructorsOf(canonical)};
  if (!constructors) {
    return nullptr;
  }
  llvm::SmallVector<mlir::Attribute> functors;
  for (const semantics::ConstructorTypes &constructor : *constructors) {
    llvm::SmallVector<mlir::Attribute> parts{
        builder_.getStringAttr(constructor.functor.name)};
    std::optional<HiddenFields> hidden{HiddenFieldsOf(constructor)};
    if (!hidden) {
      NotYet(at, "type_infos of type " + semantics::TypeToString(*canonical));
      return nullptr;
    }
    for (const semantics::Type *argument :
        ArgumentsOf(canonical, constructor)) {
      mlir::Attribute attribute{FieldTypeAttr(*argument, at, &*hidden)};
      if (!attribute) {
        return nullptr;
      }
      parts.push_back(attribute);
    }
    functors.push_back(builder_.getArrayAttr(parts));
  }
  return builder_.getArrayAttr(functors);
}

mlir::Value MerkGenerator::TypeInfoOfType(const semantics::Type *type,
    mlir::Location loc, parser::CharBlock at, bool freeIsVoid) {
  if (type->kind == semantics::TypeKind::Variable) {
    mlir::Value info{typeInfos_.lookup(type->var)};
    if (!info && freeIsVoid) {
      // Nothing determines it, as the elements of [] that nothing else uses.
      EnsureVoidDescriptor();
      return merk::TypeInfoOp::create(builder_, loc,
          merk::TypeInfoType::get(&context_), "void/0", mlir::ValueRange{});
    }
    if (!info) {
      NotYet(at, "type_infos of type " + semantics::TypeToString(*type));
    }
    return info;
  }
  std::optional<std::string> constructor{TypeConstructorName(*type)};
  if (!constructor || !EnsureDescriptor(*type, at)) {
    if (!constructor) {
      NotYet(at, "type_infos of type " + semantics::TypeToString(*type));
    }
    return nullptr;
  }
  llvm::SmallVector<mlir::Value> arguments;
  for (const semantics::Type *argument : TypeParts(*type)) {
    mlir::Value info{TypeInfoOfType(argument, loc, at, freeIsVoid)};
    if (!info) {
      return nullptr;
    }
    arguments.push_back(info);
  }
  return merk::TypeInfoOp::create(builder_, loc,
      merk::TypeInfoType::get(&context_), *constructor, arguments);
}

mlir::Value MerkGenerator::TypeInfoValue(const semantics::TypeInfo &info,
    mlir::Location loc, parser::CharBlock at) {
  auto type{merk::TypeInfoType::get(&context_)};
  switch (info.kind) {
  case semantics::TypeInfoKind::Parameter:
    return typeInfos_.lookup(info.parameter);
  case semantics::TypeInfoKind::Constant:
    return TypeInfoOfType(info.type, loc, at, /*freeIsVoid=*/true);
  case semantics::TypeInfoKind::Void:
    EnsureVoidDescriptor();
    return merk::TypeInfoOp::create(
        builder_, loc, type, "void/0", mlir::ValueRange{});
  case semantics::TypeInfoKind::Cell: {
    std::optional<std::string> constructor{TypeConstructorName(*info.type)};
    if (!constructor) {
      NotYet(at, "type_infos of type " + semantics::TypeToString(*info.type));
      return nullptr;
    }
    if (!EnsureDescriptor(*info.type, at)) {
      return nullptr;
    }
    llvm::SmallVector<mlir::Value> arguments;
    for (const semantics::TypeInfo *argument : info.arguments) {
      mlir::Value value{TypeInfoValue(*argument, loc, at)};
      if (!value) {
        return nullptr;
      }
      arguments.push_back(value);
    }
    return merk::TypeInfoOp::create(builder_, loc, type, *constructor, arguments);
  }
  case semantics::TypeInfoKind::Dictionary: {
    mlir::Value dictionary{DictionaryValue(*info.dictionary, loc, at)};
    if (!dictionary) {
      return nullptr;
    }
    return merk::TypeclassInfoWordOp::create(builder_, loc, type, dictionary,
        info.index);
  }
  case semantics::TypeInfoKind::Argument: {
    mlir::Value from{TypeInfoValue(*info.arguments.front(), loc, at)};
    if (!from) {
      return nullptr;
    }
    const semantics::Type *whole{info.arguments.front()->type};
    // A tuple's cell has the number of its arguments in word 1.
    unsigned first{whole && HasVariableArity(*whole) ? 2u : 1u};
    return merk::TypeInfoWordOp::create(
        builder_, loc, type, from, first + info.index - 1);
  }
  case semantics::TypeInfoKind::Output: {
    mlir::Value value{Given(&info)};
    if (!value) {
      NotYet(at, "a type_info given where it is not available");
    }
    return value;
  }
  }
  llvm_unreachable("a type_info's kind");
}

mlir::Value MerkGenerator::DictionaryValue(const semantics::Dictionary &dictionary,
    mlir::Location loc, parser::CharBlock at) {
  auto type{merk::TypeclassInfoType::get(&context_)};
  if (builder_.getInsertionBlock()
          ->getParentOp()
          ->getParentOfType<mlir::omp::TargetOp>() ||
      mlir::isa<mlir::omp::TargetOp>(
          builder_.getInsertionBlock()->getParentOp())) {
    // Temporary (specialization, design document, "What polymorphism costs"):
    // a device has no run-time dispatch; specialization will remove these.
    NotYet(at, "type class constraints in target regions");
    return nullptr;
  }
  switch (dictionary.kind) {
  case semantics::DictionaryKind::Parameter:
    return dictionaries_.lookup(dictionary.parameter);
  case semantics::DictionaryKind::Superclass: {
    mlir::Value from{DictionaryValue(*dictionary.from, loc, at)};
    if (!from) {
      return nullptr;
    }
    return merk::TypeclassInfoWordOp::create(
        builder_, loc, type, from, dictionary.index);
  }
  case semantics::DictionaryKind::Constant:
  case semantics::DictionaryKind::Instance: {
    if (!dictionary.instance) {
      NotYet(at, "this typeclass_info");
      return nullptr;
    }
    llvm::SmallVector<mlir::Value> arguments;
    for (const semantics::Dictionary *super : dictionary.supers) {
      arguments.push_back(DictionaryValue(*super, loc, at));
    }
    for (const semantics::TypeInfo *info : dictionary.types) {
      arguments.push_back(TypeInfoValue(*info, loc, at));
    }
    for (const semantics::TypeInfo *info : dictionary.typeInfos) {
      arguments.push_back(TypeInfoValue(*info, loc, at));
    }
    for (const semantics::Dictionary *premise : dictionary.premises) {
      arguments.push_back(DictionaryValue(*premise, loc, at));
    }
    if (llvm::is_contained(arguments, mlir::Value{})) {
      return nullptr;
    }
    return merk::TypeclassInfoOp::create(builder_, loc, type,
        InstanceSymbol(*dictionary.instance), arguments);
  }
  case semantics::DictionaryKind::Output: {
    mlir::Value value{Given(&dictionary)};
    if (!value) {
      NotYet(at, "a typeclass_info given where it is not available");
    }
    return value;
  }
  }
  llvm_unreachable("a dictionary's kind");
}

merk::ProcOp MerkGenerator::MethodProc(const ProcedureBody &body) {
  if (merk::ProcOp proc{methodProcs_.lookup(body.method)}) {
    return proc;
  }
  const Procedure &procedure{*body.procedure};
  parser::CharBlock at{procedure.symbol->source};
  std::optional<parser::Determinism> determinism{
      determinism_.DeterminismOf(procedure)};
  std::optional<merk::Determinism> merkDeterminism;
  if (determinism) {
    merkDeterminism = ClosureDeterminism(*determinism);
  }
  if (!merkDeterminism) {
    NotYet(at, "methods of this determinism");
    return nullptr;
  }
  llvm::SmallVector<mlir::Type> hidden;
  const auto *polymorphism{polymorphism_.lookup(&body)};
  if (!polymorphism) {
    NotYet(at, "this method's type_infos");
    return nullptr;
  }
  for (const semantics::HiddenArgument &argument : polymorphism->hidden) {
    hidden.push_back(argument.typeInfo
            ? mlir::Type{merk::TypeInfoType::get(&context_)}
            : mlir::Type{merk::TypeclassInfoType::get(&context_)});
  }
  std::string name{MangledName(procedure) + "$" +
      body.method->method->instanceText};
  merk::ProcOp proc{NewProc(name, at, hidden, procedure.modes,
      body.method->method->signature.arguments, *merkDeterminism)};
  if (proc) {
    methodProcs_[body.method] = proc;
  }
  return proc;
}

const ProcedureBody *MerkGenerator::MethodBody(
    const semantics::InstanceHead &head, const Procedure &procedure) {
  for (const ProcedureBody &body : determinism_.bodies()) {
    if (body.method && body.procedure == &procedure &&
        body.method->method->instance == head.item->decl) {
      return &body;
    }
  }
  return nullptr;
}

void MerkGenerator::GenerateInstances() {
  for (const semantics::InstanceHead &head : types_.instances()) {
    if (!head.item->local || head.item->decl->abstract) {
      continue;
    }
    std::string symbol{InstanceSymbol(head)};
    llvm::SmallVector<mlir::Attribute> table;
    for (const Procedure *procedure : MethodSlots(*head.typeclass)) {
      const ProcedureBody *found{MethodBody(head, *procedure)};
      parser::CharBlock at{procedure->symbol->source};
      if (!found) {
        NotYet(at, "instances without this method's body");
        continue;
      }
      std::string name{MangledName(*procedure) + "$" +
          found->method->method->instanceText + "$dispatch"};
      GenerateDispatch(head, *procedure, *found, name);
      table.push_back(mlir::FlatSymbolRefAttr::get(&context_, name));
    }
    // An instance declared in the interface is for other modules. An abstract
    // declaration has no head of its own, so any of the type class's counts.
    bool exported{head.item->exported};
    for (const semantics::InstanceItem &item : resolution_.instances()) {
      exported = exported || (item.local && item.exported && item.decl->abstract);
    }
    auto op{module_.lookupSymbol<merk::InstanceOp>(symbol)};
    op.setMethodsAttr(builder_.getArrayAttr(table));
    op.setVisibility(exported ? mlir::SymbolTable::Visibility::Public
                              : mlir::SymbolTable::Visibility::Private);
  }
}

void MerkGenerator::GenerateDispatch(const semantics::InstanceHead &head,
    const Procedure &procedure, const ProcedureBody &body,
    llvm::StringRef name) {
  parser::CharBlock at{procedure.symbol->source};
  mlir::Location loc{Location(at)};
  failed_ = false;
  // Temporary (chunk 1 step 4, docs/INPROGRESS.md): polymorphism analysis does
  // not analyse methods with existential types, nor their instances.
  if (!HiddenOutputs(procedure).empty()) {
    NotYet(at, "methods with existential types");
    return;
  }
  merk::ProcOp method{MethodProc(body)};
  const semantics::Signature *declared{types_.SignatureOf(*procedure.symbol)};
  if (!method || !declared) {
    return;
  }
  // As the type class declares it: the typeclass_info, then the method's own
  // hidden arguments, which follow the class's constraint.
  std::optional<llvm::SmallVector<mlir::Type>> hidden{
      HiddenInputs(procedure, at)};
  if (!hidden) {
    return;
  }
  llvm::SmallVector<mlir::Type> own;
  auto dictionaryType{merk::TypeclassInfoType::get(&context_)};
  own.push_back(dictionaryType);
  bool skipped{false};
  for (mlir::Type type : *hidden) {
    if (!skipped && type == dictionaryType) {
      skipped = true; // the class's own, which comes first
      continue;
    }
    own.push_back(type);
  }
  merk::ProcOp proc{NewProc(name, at, own, procedure.modes, declared->arguments,
      method.getDeterminism())};
  if (!proc) {
    return;
  }
  mlir::Block *entry{proc.addEntryBlock()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(entry);
  mlir::Value dictionary{entry->getArgument(0)};
  // The instance's hidden arguments, from the typeclass_info's cell.
  unsigned word{1 +
      static_cast<unsigned>(types_.SuperclassesOf(*head.typeclass).size() +
          head.types.size())};
  unsigned instanceVars{0};
  for (const semantics::Type *type : head.types) {
    semantics::ForEachVariable(*type,
        [&](unsigned var) { instanceVars = std::max(instanceVars, var + 1); });
  }
  llvm::SmallVector<mlir::Value> inputs;
  for (bool constrained : ConstrainedVariables(instanceVars, head.constraints)) {
    if (!constrained) {
      inputs.push_back(merk::TypeclassInfoWordOp::create(builder_, loc,
          merk::TypeInfoType::get(&context_), dictionary, word++));
    }
  }
  for (std::size_t i{0}; i < head.constraints.size(); ++i) {
    inputs.push_back(merk::TypeclassInfoWordOp::create(
        builder_, loc, dictionaryType, dictionary, word++));
  }
  unsigned input{1};
  for (; input < own.size(); ++input) {
    inputs.push_back(entry->getArgument(input));
  }
  // The arguments: a type variable's word where the instance has a type.
  llvm::ArrayRef<const semantics::Type *> instance{
      body.method->method->signature.arguments};
  llvm::SmallVector<std::pair<bool, mlir::Type>> outputs;
  for (auto [mode, classType, instanceType] :
      llvm::zip(procedure.modes, declared->arguments, instance)) {
    Represented asDeclared{*Represent(classType)};
    Represented asInstance{*Represent(instanceType)};
    if (asDeclared.dummy || IsUnused(mode)) {
      continue;
    }
    if (IsOutput(mode)) {
      outputs.push_back({asInstance.dummy, asDeclared.type});
    } else {
      mlir::Value value{entry->getArgument(input++)};
      if (!asInstance.dummy) {
        inputs.push_back(Coerce(value, asInstance.type, loc));
      }
    }
  }
  // A nondet or multi method's solutions are the dispatch's, as declared.
  mlir::ValueRange given;
  if (method.getDeterminism() == merk::Determinism::Nondet ||
      method.getDeterminism() == merk::Determinism::Multi) {
    auto call{merk::NondetCallOp::create(
        builder_, loc, method.getSymName(), inputs)};
    mlir::Block &continuation{call.getContinuation().emplaceBlock()};
    for (mlir::Type type : method.getFunctionType().getResults()) {
      continuation.addArgument(type, loc);
    }
    builder_.setInsertionPointToStart(&continuation);
    given = continuation.getArguments();
  } else {
    given = merk::CallOp::create(builder_, loc,
        method.getFunctionType().getResults(), method.getSymName(), inputs)
                .getResults();
  }
  llvm::SmallVector<mlir::Value> results;
  unsigned result{0};
  for (auto [dummy, type] : outputs) {
    if (dummy) {
      results.push_back(merk::ToWordOp::create(builder_, loc,
          merk::WordType::get(&context_),
          IntegerConstant(builder_.getI64Type(), 0, loc)));
    } else {
      results.push_back(Coerce(given[result++], type, loc));
    }
  }
  merk::SucceedOp::create(builder_, loc, results);
}

} // namespace merkurius::codegen::mlirgen
