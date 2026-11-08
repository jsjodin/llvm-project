//===-- lib/CodeGen/MLIRGen.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <set>
#include "MLIRGenImpl.h"

namespace merkurius::codegen {

static void Mangle(llvm::raw_ostream &os, llvm::StringRef name) {
  for (char c : name) {
    if (llvm::isAlnum(c) || c == '_') {
      os << c;
    } else {
      os << '$' << llvm::format_hex_no_prefix(
                       static_cast<unsigned char>(c), 2, /*Upper=*/false);
    }
  }
}

std::string MangledName(const Procedure &procedure) {
  const semantics::Symbol &symbol{*procedure.symbol};
  std::string result;
  llvm::raw_string_ostream os{result};
  llvm::SmallVector<llvm::StringRef> parts;
  symbol.module.split(parts, '.');
  for (llvm::StringRef part : parts) {
    Mangle(os, part);
    os << '.';
  }
  Mangle(os, symbol.name);
  os << '_' << (symbol.kind == semantics::SymbolKind::Function ? "f" : "")
     << symbol.arity << '_' << procedure.number;
  return result;
}

namespace mlirgen {

// The C function of a procedure's foreign_proc: merk_foreign_, then its
// symbol with `_` doubled, `.` as `_D` and `$` as `_S`.
static std::string ForeignName(const Procedure &procedure) {
  std::string result{"merk_foreign_"};
  for (char c : MangledName(procedure)) {
    result += c == '_' ? "__" : c == '.' ? "_D" : c == '$' ? "_S"
                                                            : std::string(1, c);
  }
  return result;
}

std::optional<llvm::UTF32> OnlyCodePoint(llvm::StringRef text) {
  if (text.empty()) {
    return std::nullopt;
  }
  const auto *source{reinterpret_cast<const llvm::UTF8 *>(text.data())};
  llvm::UTF32 codePoint;
  if (llvm::convertUTF8Sequence(&source, source + text.size(), &codePoint,
          llvm::strictConversion) != llvm::conversionOK ||
      source != reinterpret_cast<const llvm::UTF8 *>(text.end())) {
    return std::nullopt;
  }
  return codePoint;
}

std::optional<llvm::APInt> IntegerValue(
    llvm::StringRef text, unsigned bits) {
  bool negative{text.consume_front("-")};
  llvm::APInt value;
  if (text.consume_front("0'")) {
    std::optional<llvm::UTF32> codePoint{
        OnlyCodePoint(parser::UnquoteSpelling(("'" + text + "'").str()))};
    if (!codePoint) {
      return std::nullopt;
    }
    value = llvm::APInt{bits, *codePoint};
  } else {
    unsigned radix{10};
    if (text.consume_front("0b")) {
      radix = 2;
    } else if (text.consume_front("0o")) {
      radix = 8;
    } else if (text.consume_front("0x")) {
      radix = 16;
    }
    // No digit is i or u.
    text = text.take_until([](char c) { return c == 'i' || c == 'u'; });
    std::string digits{text};
    llvm::erase(digits, '_');
    if (llvm::StringRef{digits}.getAsInteger(radix, value)) {
      return std::nullopt;
    }
    value = value.zextOrTrunc(bits);
  }
  if (negative) {
    value.negate();
  }
  return value;
}

// The C type of a Mercury type's values (reference manual, "C data passing
// conventions"): a tuple is an MR_Tuple, a type without its own is an MR_Word,
// and an array is an MR_ArrayPtr, as mmc passes it.
static std::string CType(const semantics::Type &type) {
  if (type.kind == semantics::TypeKind::Variable) {
    return "MR_Word";
  }
  if (type.kind == semantics::TypeKind::Tuple) {
    return "MR_Tuple";
  }
  if (type.kind == semantics::TypeKind::HigherOrder) {
    return "MR_Word";
  }
  llvm::StringRef name{type.constructor->name};
  if (!type.constructor->module.empty()) {
    return type.constructor->module == "array" && name == "array"
        ? "MR_ArrayPtr"
        : "MR_Word";
  }
  if (name == "int") {
    return "MR_Integer";
  }
  if (name == "uint") {
    return "MR_Unsigned";
  }
  if (name == "float") {
    return "MR_Float";
  }
  if (name == "character") {
    return "MR_Char";
  }
  if (name == "string") {
    return "MR_String";
  }
  return (name + "_t").str(); // intN and uintN
}

// The start of the C file of a module's foreign code: the C types of the
// manual's "C data passing conventions".
static constexpr llvm::StringLiteral kForeignPrelude{R"(/* Foreign code, written by merkc. */
#include <stdint.h>
typedef intptr_t MR_Integer;
typedef uintptr_t MR_Unsigned;
typedef double MR_Float;
typedef int32_t MR_Char;
typedef char *MR_String;
typedef const char *MR_ConstString;
typedef uintptr_t MR_Word;
typedef void *MR_Tuple;
typedef void *MR_ArrayPtr;
typedef int MR_bool;
#define MR_TRUE 1
#define MR_FALSE 0
static inline MR_Float MR_word_to_float(MR_Word word) {
  MR_Float number;
  __builtin_memcpy(&number, &word, sizeof number);
  return number;
}
static inline MR_Word MR_float_to_word(MR_Float number) {
  MR_Word word;
  __builtin_memcpy(&word, &number, sizeof word);
  return word;
}
#define MR_word_to_int64(word) ((int64_t)(word))
#define MR_word_to_uint64(word) ((uint64_t)(word))
#define MR_int64_to_word(value) ((MR_Word)(value))
#define MR_uint64_to_word(value) ((MR_Word)(value))
)"};

mlir::Location MerkGenerator::Location(parser::CharBlock at) {
  if (auto range{sources_.GetProvenanceRange(at)}) {
    if (auto pos{sources_.GetSourcePosition(range->start())}) {
      return mlir::FileLineColLoc::get(&context_, *pos->path, pos->line,
          pos->column);
    }
  }
  return builder_.getUnknownLoc();
}

std::optional<Represented> MerkGenerator::Represent(
    const semantics::Type *type) {
  // A type that a goal gave is a type variable to the code that has it.
  if (type->kind == semantics::TypeKind::Variable ||
      type->kind == semantics::TypeKind::Existential) {
    return Represented{merk::WordType::get(&context_)};
  }
  if (type->kind == semantics::TypeKind::Tuple) {
    return RepresentTerm(type);
  }
  if (type->kind == semantics::TypeKind::HigherOrder) {
    return Represented{merk::ClosureType::get(&context_)};
  }
  if (type->kind != semantics::TypeKind::Named) {
    return std::nullopt;
  }
  const semantics::Symbol &constructor{*type->constructor};
  llvm::StringRef name{constructor.name};
  if (constructor.module.empty()) {
    // The primitive types (reference manual, "Primitive types"); int, uint
    // and float are a word wide (design document, "Type representation").
    if (name == "int" || name == "uint") {
      return Represented{builder_.getIntegerType(target_.WordBits())};
    }
    for (unsigned bits : {8, 16, 32, 64}) {
      if (name == "int" + std::to_string(bits) ||
          name == "uint" + std::to_string(bits)) {
        return Represented{builder_.getIntegerType(bits)};
      }
    }
    if (name == "float") {
      return Represented{builder_.getF64Type()};
    }
    if (name == "character") {
      return Represented{builder_.getIntegerType(32)};
    }
    if (name == "string") {
      return Represented{merk::StringType::get(&context_)};
    }
    return std::nullopt;
  }
  // The state of the world has no values to pass (design document,
  // "Calling convention").
  if (constructor.module == "io" && name == "state") {
    return Represented{nullptr, /*dummy=*/true};
  }
  // An array of elements a word wide at most, each in a word (design
  // document, "Type representation").
  if (constructor.module == "array" && name == "array" &&
      type->arguments.size() == 1) {
    std::optional<Represented> element{Represent(type->arguments[0])};
    if (!element || element->dummy ||
        (!mlir::isa<merk::StringType, merk::ArrayType, merk::TermType,
             merk::WordType>(
             element->type) &&
            element->type.getIntOrFloatBitWidth() > target_.WordBits())) {
      return std::nullopt;
    }
    return Represented{merk::ArrayType::get(&context_, element->type)};
  }
  return RepresentTerm(type);
}

std::optional<Represented> MerkGenerator::RepresentTerm(
    const semantics::Type *type) {
  std::string name{semantics::TypeToString(*type)};
  auto term{merk::TermType::get(&context_, name)};
  if (auto it{terms_.find(name)}; it != terms_.end()) {
    return it->second ? std::make_optional(Represented{term}) : std::nullopt;
  }
  const parser::DuTypeDefn *definition{nullptr};
  if (type->kind == semantics::TypeKind::Named) {
    if (const auto *du{std::get_if<const parser::DuTypeDefn *>(
            &type->constructor->declaration)}) {
      definition = *du;
    }
  }
  const std::vector<semantics::ConstructorTypes> *constructors{
      modes_.ops().ConstructorsOf(type)};
  // An abstract type of another module, whose constructors this module
  // cannot see, has no layout here: its values are pointers, which the
  // module that defines it makes sure they can be (Lowering).
  if (!constructors && type->kind == semantics::TypeKind::Named &&
      type->constructor->module != moduleName_ &&
      std::holds_alternative<const parser::AbstractTypeDecl *>(
          type->constructor->declaration)) {
    terms_[name] = type;
    return Represented{term};
  }
  // A type too large to print has no name of its own.
  if (!constructors || name.front() == '<' ||
      (definition && (definition->attributes || definition->supertype))) {
    terms_[name] = nullptr;
    return std::nullopt;
  }
  // Recorded before its constructors' arguments, which may be of it.
  terms_[name] = type;
  llvm::SmallVector<mlir::Attribute> attributes;
  for (const semantics::ConstructorTypes &constructor : *constructors) {
    HiddenFields hidden;
    std::optional<llvm::SmallVector<mlir::Type>> fields{
        CellFields(type, constructor, hidden)};
    if (!fields) {
      terms_[name] = nullptr;
      return std::nullopt;
    }
    attributes.push_back(merk::ConstructorAttr::get(&context_,
        constructor.functor.name, *fields,
        static_cast<unsigned>(hidden.types.size())));
  }
  // At the module's start, after those made before it.
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(module_.getBody());
  for (mlir::Operation &op : *module_.getBody()) {
    if (!mlir::isa<merk::TypeDefOp>(op)) {
      break;
    }
    builder_.setInsertionPointAfter(&op);
  }
  bool abstract{type->kind == semantics::TypeKind::Named &&
      abstractTypes_.contains(type->constructor)};
  merk::TypeDefOp::create(builder_,
      abstract ? Location(type->constructor->source) : builder_.getUnknownLoc(),
      name, builder_.getArrayAttr(attributes), abstract);
  return Represented{term};
}

std::optional<merk::Determinism> MerkDeterminism(
    parser::Determinism determinism) {
  switch (determinism) {
  case parser::Determinism::Det:
    return merk::Determinism::Det;
  case parser::Determinism::CcMulti:
    return merk::Determinism::CcMulti;
  case parser::Determinism::CcNondet:
    return merk::Determinism::CcNondet;
  case parser::Determinism::Semidet:
    return merk::Determinism::Semidet;
  case parser::Determinism::Failure:
    return merk::Determinism::Failure;
  case parser::Determinism::Erroneous:
    return merk::Determinism::Erroneous;
  default:
    return std::nullopt;
  }
}

std::optional<merk::Determinism> ClosureDeterminism(
    parser::Determinism determinism) {
  if (determinism == parser::Determinism::Nondet) {
    return merk::Determinism::Nondet;
  }
  if (determinism == parser::Determinism::Multi) {
    return merk::Determinism::Multi;
  }
  return MerkDeterminism(determinism);
}

llvm::SmallVector<bool> ConstrainedVariables(
    unsigned count, llvm::ArrayRef<semantics::ClassConstraint> constraints) {
  llvm::SmallVector<bool> constrained(count, false);
  for (const semantics::ClassConstraint &constraint : constraints) {
    for (const semantics::Type *type : constraint.arguments) {
      semantics::ForEachVariable(*type, [&](unsigned var) {
        if (var < count) {
          constrained[var] = true;
        }
      });
    }
  }
  return constrained;
}

bool HasGivenType(const semantics::Type &type) {
  bool given{false};
  semantics::ForEachExistential(type, [&](const semantics::Type &) { given = true; });
  return given;
}

std::optional<llvm::SmallVector<mlir::Type>> MerkGenerator::HiddenInputs(
    const Procedure &procedure, parser::CharBlock at) {
  llvm::SmallVector<mlir::Type> hidden;
  const semantics::Signature *signature{types_.SignatureOf(*procedure.symbol)};
  if (!signature) {
    return hidden;
  }
  // As polymorphism analysis: each of the signature's universally quantified
  // type variables, whether or not an argument has it, as a loop's predicate's
  // can have one only its clause's other goals use.
  llvm::SmallVector<bool> constrained{
      ConstrainedVariables(signature->typeVars, signature->constraints)};
  for (unsigned var{0}; var < signature->typeVars - signature->existentials;
       ++var) {
    if (!constrained[var]) {
      hidden.push_back(merk::TypeInfoType::get(&context_));
    }
  }
  hidden.append(signature->constraints.size(),
      merk::TypeclassInfoType::get(&context_));
  return hidden;
}

llvm::SmallVector<mlir::Type> MerkGenerator::HiddenOutputs(
    const Procedure &procedure) {
  llvm::SmallVector<mlir::Type> hidden;
  const semantics::Signature *signature{types_.SignatureOf(*procedure.symbol)};
  if (!signature) {
    return hidden;
  }
  llvm::SmallVector<bool> constrained{ConstrainedVariables(
      signature->typeVars, signature->existentialConstraints)};
  for (unsigned var{signature->typeVars - signature->existentials};
       var < signature->typeVars; ++var) {
    if (!constrained[var]) {
      hidden.push_back(merk::TypeInfoType::get(&context_));
    }
  }
  hidden.append(signature->existentialConstraints.size(),
      merk::TypeclassInfoType::get(&context_));
  return hidden;
}

std::optional<std::pair<unsigned, unsigned>> MerkGenerator::ConstrainedPlace(
    unsigned var, llvm::ArrayRef<semantics::ClassConstraint> constraints,
    unsigned first) {
  for (auto [i, constraint] : llvm::enumerate(constraints)) {
    unsigned supers{static_cast<unsigned>(
        types_.SuperclassesOf(*constraint.typeclass).size())};
    for (auto [j, argument] : llvm::enumerate(constraint.arguments)) {
      if (argument->kind == semantics::TypeKind::Variable &&
          argument->var == var) {
        return std::make_pair(first + static_cast<unsigned>(i),
            supers + static_cast<unsigned>(j) + 1);
      }
    }
  }
  return std::nullopt;
}

std::optional<MerkGenerator::HiddenFields> MerkGenerator::HiddenFieldsOf(
    const semantics::ConstructorTypes &constructor) {
  HiddenFields hidden;
  const semantics::Signature *signature{constructor.symbol
          ? types_.SignatureOf(*constructor.symbol)
          : nullptr};
  if (!signature || signature->existentials == 0) {
    return hidden;
  }
  // As polymorphism analysis passes them to a construction: the type_infos
  // of the variables the constraints do not have, then the typeclass_infos.
  unsigned universal{signature->typeVars - signature->existentials};
  llvm::ArrayRef<semantics::ClassConstraint> constraints{
      signature->existentialConstraints};
  llvm::SmallVector<bool> constrained{
      ConstrainedVariables(signature->typeVars, constraints)};
  hidden.places.resize(signature->existentials);
  for (unsigned var{universal}; var < signature->typeVars; ++var) {
    if (!constrained[var]) {
      hidden.places[var - universal] = {
          static_cast<unsigned>(hidden.types.size()), 0};
      hidden.types.push_back(merk::TypeInfoType::get(&context_));
    }
  }
  unsigned first{static_cast<unsigned>(hidden.types.size())};
  hidden.types.append(
      constraints.size(), merk::TypeclassInfoType::get(&context_));
  // A constrained variable's type_info is in the typeclass_info of the first
  // constraint that has it as an argument, after the method table and the
  // superclasses' typeclass_infos.
  for (unsigned var{universal}; var < signature->typeVars; ++var) {
    if (!constrained[var]) {
      continue;
    }
    std::optional<std::pair<unsigned, unsigned>> place{
        ConstrainedPlace(var, constraints, first)};
    if (!place) {
      return std::nullopt;
    }
    hidden.places[var - universal] = *place;
  }
  return hidden;
}

unsigned MerkGenerator::ExistentialBase(const semantics::Type &type) {
  unsigned base{kCellExistential};
  semantics::ForEachVariable(
      type, [&](unsigned var) { base = std::max(base, var + 1); });
  return base;
}

llvm::SmallVector<const semantics::Type *> MerkGenerator::ArgumentsOf(
    const semantics::Type *type, const semantics::ConstructorTypes &constructor) {
  const semantics::Signature *signature{constructor.symbol
          ? types_.SignatureOf(*constructor.symbol)
          : nullptr};
  if (!signature || signature->existentials == 0) {
    return {constructor.arguments.begin(), constructor.arguments.end()};
  }
  unsigned universal{signature->typeVars - signature->existentials};
  unsigned base{ExistentialBase(*type)};
  semantics::TypeContext &context{types_.types()};
  llvm::SmallVector<const semantics::Type *> arguments;
  for (const semantics::Type *argument : signature->arguments.drop_back()) {
    arguments.push_back(context.Substitute(argument, [&](unsigned var) {
      if (var >= universal) {
        return context.Variable(base + var - universal);
      }
      return var < type->arguments.size() ? type->arguments[var]
                                          : context.Variable(var);
    }));
  }
  return arguments;
}

std::optional<llvm::SmallVector<mlir::Type>> MerkGenerator::CellFields(
    const semantics::Type *type, const semantics::ConstructorTypes &constructor,
    HiddenFields &hidden) {
  std::optional<HiddenFields> found{HiddenFieldsOf(constructor)};
  if (!found) {
    return std::nullopt;
  }
  hidden = std::move(*found);
  llvm::SmallVector<mlir::Type> fields{hidden.types};
  for (const semantics::Type *argument : ArgumentsOf(type, constructor)) {
    std::optional<Represented> represented{Represent(argument)};
    if (!represented || represented->dummy) {
      return std::nullopt;
    }
    fields.push_back(represented->type);
  }
  return fields;
}

const semantics::ConstructorTypes *MerkGenerator::ConstructorOf(
    const semantics::Type *type, llvm::StringRef name, unsigned arity) {
  const std::vector<semantics::ConstructorTypes> *constructors{
      modes_.ops().ConstructorsOf(type)};
  if (!constructors) {
    return nullptr;
  }
  for (const semantics::ConstructorTypes &constructor : *constructors) {
    if (constructor.functor.name == name && constructor.functor.arity == arity) {
      return &constructor;
    }
  }
  return nullptr;
}

mlir::Value MerkGenerator::HiddenTypeInfo(mlir::ValueRange fields,
    std::pair<unsigned, unsigned> place, mlir::Location loc) {
  mlir::Value field{fields[place.first]};
  if (place.second == 0) {
    return field;
  }
  return merk::TypeclassInfoWordOp::create(builder_, loc,
      merk::TypeInfoType::get(&context_), field, place.second);
}

mlir::Value MerkGenerator::Given(const void *node) {
  auto it{givenVars_.find(node)};
  return it == givenVars_.end() ? mlir::Value{} : values_.lookup(it->second);
}

void MerkGenerator::NoteGivenType(const semantics::TypeInfo &info) {
  if (info.type && info.type->kind == semantics::TypeKind::Variable) {
    givenTypes_[info.type->var] = {&info, 0};
  }
}

void MerkGenerator::NoteGivenTypes(const semantics::Dictionary &dictionary) {
  unsigned supers{static_cast<unsigned>(
      types_.SuperclassesOf(*dictionary.constraint.typeclass).size())};
  for (auto [j, argument] : llvm::enumerate(dictionary.constraint.arguments)) {
    if (argument->kind == semantics::TypeKind::Variable) {
      givenTypes_.try_emplace(argument->var, &dictionary,
          supers + static_cast<unsigned>(j) + 1);
    }
  }
}

static unsigned GivenKey(const semantics::Type &type) {
  return type.kind == semantics::TypeKind::Existential ? kGivenTypeKey + type.var
                                                       : type.var;
}

// Whether a type has a type variable or a type a goal gave.
static bool HasTypeVariable(const semantics::Type &type) {
  if (type.kind == semantics::TypeKind::Variable ||
      type.kind == semantics::TypeKind::Existential) {
    return true;
  }
  return (type.kind == semantics::TypeKind::Named ||
             type.kind == semantics::TypeKind::Tuple) &&
      llvm::any_of(type.arguments, [](const semantics::Type *argument) {
        return HasTypeVariable(*argument);
      });
}

void MerkGenerator::GiveTypeInfo(
    const semantics::Type &type, mlir::Value info, mlir::Location loc) {
  if (type.kind == semantics::TypeKind::Variable ||
      type.kind == semantics::TypeKind::Existential) {
    givenInfos_[GivenKey(type)] = info;
    return;
  }
  if (type.kind != semantics::TypeKind::Named &&
      type.kind != semantics::TypeKind::Tuple) {
    return;
  }
  // Word 0 of a cell is the descriptor, and a tuple's word 1 its arity.
  std::int64_t first{type.kind == semantics::TypeKind::Tuple ? 2 : 1};
  for (auto [k, argument] : llvm::enumerate(type.arguments)) {
    if (HasTypeVariable(*argument)) {
      GiveTypeInfo(*argument,
          merk::TypeInfoWordOp::create(builder_, loc,
              merk::TypeInfoType::get(&context_), info,
              first + static_cast<std::int64_t>(k)),
          loc);
    }
  }
}

void MerkGenerator::GiveTypeInfo(
    const semantics::TypeInfo &info, mlir::Value value, mlir::Location loc) {
  if (info.type) {
    GiveTypeInfo(*info.type, value, loc);
  }
}

void MerkGenerator::GiveDictionary(const semantics::Dictionary &dictionary,
    mlir::Value value, mlir::Location loc) {
  unsigned supers{static_cast<unsigned>(
      types_.SuperclassesOf(*dictionary.constraint.typeclass).size())};
  for (auto [j, argument] : llvm::enumerate(dictionary.constraint.arguments)) {
    if (HasTypeVariable(*argument)) {
      GiveTypeInfo(*argument,
          merk::TypeclassInfoWordOp::create(builder_, loc,
              merk::TypeInfoType::get(&context_), value,
              supers + static_cast<unsigned>(j) + 1),
          loc);
    }
  }
}

void MerkGenerator::Give(const void *node, mlir::Value value) {
  auto [it, added]{givenVars_.try_emplace(
      node, kFirstGivenVar + static_cast<BodyVar>(givenVars_.size()))};
  values_[it->second] = value;
}

bool MerkGenerator::CanMake(const semantics::TypeInfo &info) {
  switch (info.kind) {
  case semantics::TypeInfoKind::Output:
    return static_cast<bool>(Given(&info));
  case semantics::TypeInfoKind::Dictionary:
    return CanMake(*info.dictionary);
  default:
    return llvm::all_of(info.arguments,
        [&](const semantics::TypeInfo *argument) { return CanMake(*argument); });
  }
}

bool MerkGenerator::CanMake(const semantics::Dictionary &dictionary) {
  switch (dictionary.kind) {
  case semantics::DictionaryKind::Output:
    return static_cast<bool>(Given(&dictionary));
  case semantics::DictionaryKind::Superclass:
    return CanMake(*dictionary.from);
  default:
    return CanMakeAll(dictionary.typeInfos) && CanMakeAll(dictionary.types) &&
        CanMakeAll(dictionary.premises) && CanMakeAll(dictionary.supers);
  }
}

llvm::SmallVector<const Procedure *> MerkGenerator::MethodSlots(
    const semantics::Symbol &typeclass) {
  llvm::SmallVector<const Procedure *> slots;
  const semantics::ModuleSymbols *symbols{typeclass.module == moduleName_
          ? &resolution_.module()
          : resolution_.Module(typeclass.module)};
  if (!symbols) {
    return slots;
  }
  for (const semantics::Symbol *symbol : symbols->symbols()) {
    if (symbol->owner == &typeclass) {
      llvm::ArrayRef<const Procedure *> procedures{modes_.ProceduresOf(*symbol)};
      slots.append(procedures.begin(), procedures.end());
    }
  }
  return slots;
}

std::string MerkGenerator::InstanceSymbol(
    const semantics::InstanceHead &head) {
  std::string text;
  llvm::raw_string_ostream os{text};
  os << "merk.instance." << head.typeclass->module << '.'
     << head.typeclass->name << '(';
  llvm::ListSeparator separator{","};
  for (const semantics::Type *type : head.types) {
    os << separator
       << semantics::TypeToString(*type,
              [](unsigned var) { return "V" + std::to_string(var); });
  }
  os << ')';
  if (!module_.lookupSymbol<merk::InstanceOp>(text)) {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToStart(module_.getBody());
    merk::InstanceOp::create(builder_, builder_.getUnknownLoc(),
        builder_.getStringAttr(text), /*sym_visibility=*/mlir::StringAttr{},
        /*methods=*/mlir::ArrayAttr{}, /*shape=*/mlir::DenseI64ArrayAttr{});
  }
  return text;
}

mlir::Value MerkGenerator::Coerce(
    mlir::Value value, mlir::Type type, mlir::Location loc) {
  if (value.getType() == type) {
    return value;
  }
  auto word{merk::WordType::get(&context_)};
  // A value back from its word is the word's, and a word back to the type of
  // the value it came from is the value, in the same region.
  if (mlir::Operation *from{value.getDefiningOp()};
      from && from->getParentRegion() == builder_.getInsertionBlock()->getParent()) {
    if (auto fromWord{mlir::dyn_cast<merk::FromWordOp>(from)}; fromWord && type == word) {
      return fromWord.getWord();
    }
    if (auto toWord{mlir::dyn_cast<merk::ToWordOp>(from)};
        toWord && toWord.getValue().getType() == type) {
      return toWord.getValue();
    }
  }
  if (value.getType() != word) {
    value = merk::ToWordOp::create(builder_, loc, word, value);
  }
  if (type != word) {
    value = merk::FromWordOp::create(builder_, loc, type, value);
  }
  return value;
}

// The type variables of a type, in the order its name has them: a variable
// once, and each occurrence of a type that a goal gave, which has its own.
// False for a type with one the runtime cannot name.
static bool CollectTypeVariables(
    const semantics::Type *type, llvm::SmallVectorImpl<unsigned> &variables) {
  switch (type->kind) {
  case semantics::TypeKind::Variable:
    if (!llvm::is_contained(variables, type->var)) {
      variables.push_back(type->var);
    }
    return true;
  case semantics::TypeKind::Existential:
    variables.push_back(kGivenTypeKey + type->var);
    return true;
  case semantics::TypeKind::Named:
  case semantics::TypeKind::Tuple:
    for (const semantics::Type *argument : type->arguments) {
      if (!CollectTypeVariables(argument, variables)) {
        return false;
      }
    }
    return true;
  case semantics::TypeKind::HigherOrder:
    for (const semantics::Type *argument : type->arguments) {
      if (!CollectTypeVariables(argument, variables)) {
        return false;
      }
    }
    return !type->result || CollectTypeVariables(type->result, variables);
  default:
    return true;
  }
}

void MerkGenerator::AnnotateRoots(const ProcedureBody &body) {
  if (failed_ || Terminated() || target_.options().openMPIsTargetDevice) {
    return;
  }
  // A value's note is in the region that defines it. Outlining takes it along
  // with a value that it captures. In order, as the notes decide the order of
  // what outlining captures.
  mlir::Region *here{builder_.getBlock()->getParent()};
  llvm::SmallVector<BodyVar> bound;
  for (auto [var, value] : values_) {
    if (var < body.vars.size() && value && value.getParentRegion() == here &&
        mlir::isa<merk::WordType, merk::TermType, merk::ArrayType>(
            value.getType()) &&
        !annotated_.contains(value)) {
      bound.push_back(var);
    }
  }
  llvm::sort(bound);
  for (BodyVar var : bound) {
    mlir::Value value{values_.lookup(var)};
    if (annotated_.contains(value)) {
      continue;
    }
    rootTypes_.insert(body.vars[var].type);
    if (NoteRoot(value, body.vars[var].type)) {
      annotated_.insert(value);
    }
  }
}

bool MerkGenerator::NoteRoot(mlir::Value value, const semantics::Type *type) {
  if (target_.options().openMPIsTargetDevice ||
      !mlir::isa<merk::WordType, merk::TermType, merk::ArrayType>(
          value.getType())) {
    return true;
  }
  llvm::SmallVector<unsigned> variables;
  if (!CollectTypeVariables(type, variables)) {
    return false;
  }
  if (variables.empty()) {
    return true;
  }
  mlir::Region *here{builder_.getBlock()->getParent()};
  llvm::SmallVector<mlir::Value> infos;
  for (unsigned variable : variables) {
    mlir::Value info{typeInfos_.lookup(variable)};
    if (!info) {
      info = givenInfos_.lookup(variable);
    }
    if (!info && variable < kCellExistential) {
      // Nothing determines it, as the elements of a [] that nothing uses.
      // At the start, as it must be there at every safepoint the value is at.
      EnsureVoidDescriptor();
      mlir::OpBuilder::InsertionGuard guard{builder_};
      auto proc{mlir::dyn_cast<merk::ProcOp>(builder_.getBlock()->getParentOp())};
      if (!proc) {
        proc = builder_.getBlock()->getParentOp()->getParentOfType<merk::ProcOp>();
      }
      builder_.setInsertionPointToStart(&proc.getBody().front());
      info = merk::TypeInfoOp::create(builder_, value.getLoc(),
          merk::TypeInfoType::get(&context_), "void/0", mlir::ValueRange{});
    }
    if (!info || !info.getParentRegion()->isAncestor(here)) {
      return false;
    }
    infos.push_back(info);
  }
  merk::RootTypeOp::create(builder_, value.getLoc(), value, infos);
  return true;
}

void MerkGenerator::Goal(const Body &goal, const ProcedureBody &body) {
  // Before as well, as a goal that ends the body leaves no place after it.
  AnnotateRoots(body);
  GoalBody(goal, body);
  AnnotateRoots(body);
}

void MerkGenerator::GoalBody(const Body &goal, const ProcedureBody &body) {
  if (failed_ || Terminated()) {
    return;
  }
  if (committing_ != &goal && NeedsCommit(goal, body)) {
    Commit(goal, body);
    return;
  }
  switch (goal.kind) {
  case BodyKind::Disj:
    if (goal.goals.empty()) {
      merk::FailOp::create(builder_, Location(goal.tokens));
      return;
    }
    Disjunction(goal, body);
    return;
  case BodyKind::IfThenElse:
    IfThenElse(goal, body);
    return;
  case BodyKind::Switch:
    Switch(goal, body);
    return;
  case BodyKind::Not:
    Not(goal, body);
    return;
  case BodyKind::Conj:
    for (const Body *conjunct : goal.goals) {
      Goal(*conjunct, body);
    }
    return;
  case BodyKind::Scope:
    if (goal.directive) {
      Omp(goal, body);
      return;
    }
    if (goal.trace) {
      Trace(goal, body);
      return;
    }
    Goal(*goal.goals.front(), body);
    return;
  case BodyKind::Call:
    Call(goal, *goal.modes->callee, goal.arguments, body);
    return;
  case BodyKind::HOCall:
    HigherOrderCall(goal, body);
    return;
  case BodyKind::Unify:
    if (goal.switchTest) {
      // The switch's case has tested it; a constructor with arguments is
      // taken apart.
      if (!goal.isVar && !goal.arguments.empty()) {
        TermUnify(goal, body);
      }
      return;
    }
    if (goal.lambda) {
      LambdaClosure(goal, body);
      return;
    }
    if (!goal.isVar && goal.unify == UnifyKind::Construct && goal.modes &&
        goal.modes->callee &&
        body.vars[goal.left].type->kind == semantics::TypeKind::HigherOrder) {
      NameClosure(goal, body);
      return;
    }
    if (goal.isVar && goal.unify == UnifyKind::Test) {
      // X = X succeeds, even for a NaN, as with mmc.
      if (goal.left != goal.var) {
        Test(goal, ValueOf(goal.left), ValueOf(goal.var),
            body.vars[goal.left].type);
      }
      return;
    }
    if (!goal.isVar && goal.unify != UnifyKind::Call &&
        goal.unify != UnifyKind::Assign &&
        mlir::isa_and_nonnull<merk::TermType>(
            Represent(body.vars[goal.left].type).value_or(Represented{})
                .type)) {
      TermUnify(goal, body);
      return;
    }
    if (!goal.isVar && goal.arguments.empty() && goal.source &&
        (goal.unify == UnifyKind::Deconstruct ||
            goal.unify == UnifyKind::Test)) {
      if (mlir::Value constant{Constant(goal, body)}) {
        Test(goal, ValueOf(goal.left), constant, body.vars[goal.left].type);
      }
      return;
    }
    if (goal.isVar && goal.unify == UnifyKind::Assign) {
      // X = Y: the variable assigned is the other's value; one of a type
      // with no values has none.
      if (values_.count(goal.var)) {
        values_[goal.left] = values_.lookup(goal.var);
      } else if (mlir::Value value{values_.lookup(goal.left)}) {
        values_[goal.var] = value;
      } else if (mlir::Value word{values_.lookup(kDummyWordVar + goal.var)}) {
        // The word of one of a type with no values, in a body that gives
        // words, goes with it.
        values_[kDummyWordVar + goal.left] = word;
      } else if (mlir::Value word{values_.lookup(kDummyWordVar + goal.left)}) {
        values_[kDummyWordVar + goal.var] = word;
      }
      return;
    }
    if (goal.unify == UnifyKind::Construct && !goal.isVar &&
        goal.arguments.empty() && goal.source) {
      if (mlir::Value value{Constant(goal, body)}) {
        values_[goal.left] = value;
      }
      return;
    }
    if (goal.unify == UnifyKind::Call && goal.modes && goal.modes->callee) {
      llvm::SmallVector<BodyVar> arguments(
          goal.arguments.begin(), goal.arguments.end());
      arguments.push_back(goal.left);
      Call(goal, *goal.modes->callee, arguments, body);
      return;
    }
    NotYet(Where(goal.tokens), "this unification");
    return;
  default:
    NotYet(Where(goal.tokens),
        "goals of kind " + std::string{semantics::EnumToString(goal.kind)});
    return;
  }
}

mlir::Value MerkGenerator::Constant(const Body &goal, const ProcedureBody &body) {
  const semantics::Type *type{body.vars[goal.left].type};
  std::optional<Represented> represented{Represent(type)};
  if (!represented || represented->dummy) {
    NotYet(Where(goal.tokens),
        "values of type " + semantics::TypeToString(*type));
    return nullptr;
  }
  mlir::Type mlirType{represented->type};
  const semantics::Functor &functor{goal.source->functor};
  mlir::TypedAttr value;
  if (functor.kind == semantics::FunctorKind::Literal) {
    llvm::StringRef text{functor.literal->text};
    switch (functor.literal->kind) {
    case parser::LiteralKind::Integer:
      if (auto integer{mlir::dyn_cast<mlir::IntegerType>(mlirType)}) {
        if (std::optional<llvm::APInt> v{
                IntegerValue(text, integer.getWidth())}) {
          value = builder_.getIntegerAttr(integer, *v);
        }
      }
      break;
    case parser::LiteralKind::Float:
      if (auto f{mlir::dyn_cast<mlir::FloatType>(mlirType)}) {
        std::string digits{text};
        llvm::erase(digits, '_');
        llvm::APFloat v{f.getFloatSemantics()};
        if (auto status{v.convertFromString(
                digits, llvm::APFloat::rmNearestTiesToEven)}) {
          value = builder_.getFloatAttr(f, v);
        } else {
          llvm::consumeError(status.takeError());
        }
      }
      break;
    case parser::LiteralKind::String:
      if (mlir::isa<merk::StringType>(mlirType)) {
        value = mlir::StringAttr::get(parser::UnquoteSpelling(text), mlirType);
      }
      break;
    }
  } else if (functor.kind == semantics::FunctorKind::Name &&
      mlirType.isInteger(32) && type->constructor->name == "character") {
    // A character constant is its code point.
    if (std::optional<llvm::UTF32> codePoint{
            OnlyCodePoint(functor.name->name)}) {
      value = builder_.getI32IntegerAttr(*codePoint);
    }
  }
  if (!value) {
    NotYet(Where(goal.tokens), "this unification");
    return nullptr;
  }
  return merk::ConstantOp::create(
      builder_, Location(goal.tokens), value.getType(), value);
}

std::string MerkGenerator::LineDirective(parser::TokenRange range) {
  parser::CharBlock at{Where(range)};
  if (auto provenance{sources_.GetProvenanceRange(at)}) {
    if (auto pos{sources_.GetSourcePosition(provenance->start())}) {
      std::string path{*pos->path};
      std::string escaped;
      for (char c : path) {
        if (c == '"' || c == '\\') {
          escaped += '\\';
        }
        escaped += c;
      }
      return "#line " + std::to_string(pos->line) + " \"" + escaped + "\"\n";
    }
  }
  return "";
}

void MerkGenerator::GenerateForeign(
    const Procedure &procedure, const parser::ForeignProcPragma &pragma) {
  parser::CharBlock at{procedure.symbol->source};
  failed_ = false;
  const semantics::Signature *signature{
      types_.SignatureOf(*procedure.symbol)};
  if (!signature) {
    return;
  }
  std::optional<llvm::SmallVector<mlir::Type>> hiddenInputs{
      HiddenInputs(procedure, at)};
  if (!hiddenInputs) {
    return;
  }
  if (!signature->constraints.empty()) {
    NotYet(at, "foreign_procs with type class constraints");
    return;
  }
  if (!HiddenOutputs(procedure).empty()) {
    NotYet(at, "foreign_procs with existential types");
    return;
  }
  std::optional<unsigned> hidden{static_cast<unsigned>(hiddenInputs->size())};
  merk::ProcOp proc{ProcFor(procedure, signature->arguments, at)};
  if (!proc) {
    return;
  }
  if (procedure.symbol->exported) {
    proc.setPublic();
  }
  // The foreign_proc's variables: its arguments', then a function's
  // result.
  llvm::SmallVector<const parser::ForeignProcArg *> arguments(
      pragma.arguments.begin(), pragma.arguments.end());
  if (pragma.result) {
    arguments.push_back(pragma.result);
  }
  // The C function takes the inputs, and returns the only output or else
  // stores the outputs through pointers after the inputs; that of a
  // procedure that can fail returns SUCCESS_INDICATOR, and stores every
  // output. A variable of a type with no values, such as io.state, is an
  // MR_Word set to 0. A type variable's value is an MR_Word, and its
  // type_info is one too, the parameter TypeInfo_for_T, first, as in mmc:
  // the manual does not describe them.
  merk::Determinism determinism{proc.getDeterminism()};
  bool canFail{determinism == merk::Determinism::Semidet ||
      determinism == merk::Determinism::Failure ||
      determinism == merk::Determinism::CcNondet};
  std::string name{ForeignName(procedure)};
  std::string parameters, locals, stores;
  if (canFail) {
    locals += "  MR_bool SUCCESS_INDICATOR = MR_TRUE;\n";
  }
  for (unsigned var{0}; var < *hidden; ++var) {
    parameters += (parameters.empty() ? "" : ", ");
    parameters += "MR_Word TypeInfo_for_" + signature->varNames[var].str();
  }
  llvm::SmallVector<std::pair<std::string, std::string>> outputs;
  for (auto [argument, type, mode] :
      llvm::zip(arguments, signature->arguments, procedure.modes)) {
    std::string var{argument->var->name};
    if (Represent(type)->dummy || IsUnused(mode)) {
      locals += "  MR_Word " + var + " = 0;\n  (void)" + var + ";\n";
    } else if (IsOutput(mode)) {
      outputs.emplace_back(CType(*type), var);
      locals += "  " + CType(*type) + " " + var + ";\n";
    } else {
      parameters += (parameters.empty() ? "" : ", ") + CType(*type) + " " + var;
    }
  }
  std::string result{"void"};
  if (outputs.size() == 1 && !canFail) {
    result = outputs.front().first;
    stores = "  return " + outputs.front().second + ";\n";
  } else {
    for (auto &[ctype, var] : outputs) {
      parameters += (parameters.empty() ? "" : ", ") + ctype + " *merk_out_" + var;
      stores += "  *merk_out_" + var + " = " + var + ";\n";
    }
    if (canFail) {
      result = "MR_bool";
      stores += "  return SUCCESS_INDICATOR;\n";
    }
  }
  foreignCode_ += "\n" + result + " " + name + "(" +
      (parameters.empty() ? "void" : parameters) + ") {\n" + locals + "  {\n" +
      LineDirective(pragma.code->tokens) + pragma.code->text.str() +
      "\n  }\n" + stores + "}\n";

  mlir::Block *entry{proc.addEntryBlock()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(entry);
  mlir::Location loc{Location(Where(pragma.tokens))};
  // A term is an MR_Word or MR_Tuple in C whatever its layout, so it crosses
  // as its word: a notag type's float would otherwise be in a float register.
  // So does a type_info.
  auto word{merk::WordType::get(&context_)};
  llvm::SmallVector<mlir::Value> inputs;
  for (mlir::Value argument : entry->getArguments()) {
    inputs.push_back(
        mlir::isa<merk::TermType, merk::TypeInfoType, merk::ClosureType>(
            argument.getType())
            ? mlir::Value{merk::ToWordOp::create(builder_, loc, word, argument)}
            : argument);
  }
  llvm::SmallVector<mlir::Type> results;
  if (canFail) {
    results.push_back(builder_.getI1Type());
  }
  for (mlir::Type type : proc.getFunctionType().getResults()) {
    results.push_back(mlir::isa<merk::TermType>(type) ? word : type);
  }
  auto call{merk::ForeignCallOp::create(
      builder_, loc, results, name, inputs, canFail)};
  mlir::ValueRange solution{call.getResults()};
  if (canFail) {
    merk::GuardOp::create(builder_, loc, solution.front());
    solution = solution.drop_front();
  }
  llvm::SmallVector<mlir::Value> values;
  for (auto [value, type] :
      llvm::zip(solution, proc.getFunctionType().getResults())) {
    values.push_back(mlir::isa<merk::TermType>(type)
            ? mlir::Value{merk::FromWordOp::create(builder_, loc, type, value)}
            : value);
  }
  merk::SucceedOp::create(builder_, loc, values);
}

void MerkGenerator::StubBody(merk::ProcOp proc, parser::CharBlock at) {
  mlir::Location loc{Location(at)};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(proc.addEntryBlock());
  auto string{merk::StringType::get(&context_)};
  mlir::Value what{merk::ConstantOp::create(builder_, loc, string,
      mlir::StringAttr::get(proc.getSymName() + ": " + notYet_, string))};
  merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
      "merk_not_implemented", mlir::ValueRange{what}, /*can_fail=*/false);
  merk::FailOp::create(builder_, loc);
}

// What mode analysis does not check yet in an unchecked procedure's clauses.
static llvm::StringRef UncheckedWhat(semantics::Unchecked reason) {
  switch (reason) {
  case semantics::Unchecked::ParallelConjunction:
    return "parallel conjunctions";
  case semantics::Unchecked::FieldClosure:
    return "closures of functions or fields";
  case semantics::Unchecked::TraceState:
    return "trace goals with state parameters";
  case semantics::Unchecked::Try:
    return "try goals";
  case semantics::Unchecked::Atomic:
    return "atomic goals";
  case semantics::Unchecked::Loop:
    return "loops";
  case semantics::Unchecked::InstParameters:
    return "lambda expressions with inst parameters";
  case semantics::Unchecked::InferredModes:
    return "calls of predicates whose modes are inferred";
  case semantics::Unchecked::TypeErrors:
  case semantics::Unchecked::None:
    break;
  }
  return "goals";
}

// A procedure whose clauses mode analysis did not check has no body to
// generate: it is reported, or with -fstub-unimplemented is a stub, rather
// than left for its callers to fail to link.
void MerkGenerator::GenerateUnchecked(const Procedure &procedure) {
  parser::CharBlock at{procedure.symbol->source};
  std::string what{("procedures with " +
      UncheckedWhat(modes_.UncheckedReason(procedure)) +
      ", which mode analysis does not check yet")
                       .str()};
  if (!stubUnimplemented_) {
    failed_ = false;
    NotYet(at, what);
    return;
  }
  const semantics::Signature *signature{types_.SignatureOf(*procedure.symbol)};
  if (!signature) {
    return;
  }
  llvm::SmallVector<const semantics::Type *> types{
      signature->arguments.begin(), signature->arguments.end()};
  // As in a body: a procedure without a signature code generation can make
  // is left out.
  failed_ = false;
  inBody_ = true;
  llvm::scope_exit leave{[&] { inBody_ = false; }};
  merk::ProcOp proc{ProcFor(procedure, types, at)};
  if (!proc || !proc.isExternal()) {
    return;
  }
  if (procedure.symbol->exported) {
    proc.setPublic();
  }
  notYet_ = "code generation for " + what;
  StubBody(proc, at);
}

// solutions.unsorted_aggregate/4, unsorted_aggregate2/6 and do_while/4,
// which have no clauses: code generation makes each of their procedures
// from merk.aggregate (design document, Lowering, "Collecting solutions").
static bool IsAggregate(const semantics::Symbol &symbol) {
  return symbol.module == "solutions" &&
      symbol.kind == semantics::SymbolKind::Predicate &&
      ((symbol.name == "unsorted_aggregate" && symbol.arity == 4) ||
          (symbol.name == "unsorted_aggregate2" && symbol.arity == 6) ||
          (symbol.name == "do_while" && symbol.arity == 4));
}

// The procedure takes the type_infos, the generator, the step and the
// accumulators' first values, and gives their last; every one of them but
// the closures is a word, of a type variable. Its body is merk.aggregate,
// whose generator calls the generator closure, giving each of its solutions,
// and whose step calls the step closure; do_while's step goes on if the
// closure says `yes`, the others' always.
void MerkGenerator::GenerateAggregate(const Procedure &procedure) {
  parser::CharBlock at{procedure.symbol->source};
  mlir::Location loc{Location(at)};
  failed_ = false;
  inBody_ = true;
  notYet_.clear();
  llvm::scope_exit leave{[&] { inBody_ = false; }};
  const semantics::Signature *signature{types_.SignatureOf(*procedure.symbol)};
  if (!signature) {
    return;
  }
  const semantics::Inst *stepInst{procedure.modes[1].initial};
  std::optional<merk::Determinism> stepDeterminism;
  if (stepInst->kind == semantics::InstKind::HigherOrder) {
    stepDeterminism = MerkDeterminism(stepInst->determinism);
  }
  if (!stepDeterminism) {
    NotYet(at, "this mode of " + procedure.symbol->name.str());
    return;
  }
  merk::ProcOp proc{ProcFor(procedure, signature->arguments, at)};
  if (!proc || !proc.isExternal()) {
    return;
  }
  if (procedure.symbol->exported) {
    proc.setPublic();
  }
  bool stops{procedure.symbol->name == "do_while"};
  auto word{merk::WordType::get(&context_)};
  unsigned count{proc.getFunctionType().getNumResults()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  mlir::Block &entry{*proc.addEntryBlock()};
  unsigned hidden{entry.getNumArguments() - 2 - count};
  mlir::Value generator{entry.getArgument(hidden)};
  mlir::Value step{entry.getArgument(hidden + 1)};
  builder_.setInsertionPointToStart(&entry);
  // The accumulators' types, for the collector. Nothing constrains these type
  // variables, so the type_infos are in the order of their numbers.
  typeInfos_.clear();
  givenInfos_.clear();
  for (unsigned variable{0}; variable < hidden; ++variable) {
    typeInfos_[variable] = entry.getArgument(variable);
  }
  unsigned accumulator{hidden + 2};
  for (auto [type, mode] : llvm::zip(
           llvm::ArrayRef(signature->arguments).drop_front(2),
           llvm::ArrayRef(procedure.modes).drop_front(2))) {
    if (!IsOutput(mode) && accumulator < entry.getNumArguments()) {
      NoteRoot(entry.getArgument(accumulator++), type);
    }
  }
  auto aggregate{merk::AggregateOp::create(
      builder_, loc, entry.getArguments().drop_front(hidden + 2))};
  builder_.setInsertionPointToStart(&aggregate.getGenerator().emplaceBlock());
  auto call{merk::NondetCallClosureOp::create(
      builder_, loc, generator, mlir::ValueRange{})};
  mlir::Block &continuation{call.getContinuation().emplaceBlock()};
  continuation.addArgument(word, loc);
  builder_.setInsertionPointToStart(&continuation);
  merk::SolutionOp::create(builder_, loc, continuation.getArguments());
  mlir::Block &body{aggregate.getStep().emplaceBlock()};
  for (unsigned i{0}; i <= count; ++i) {
    body.addArgument(word, loc);
  }
  builder_.setInsertionPointToStart(&body);
  auto apply{merk::CallClosureOp::create(builder_, loc,
      llvm::SmallVector<mlir::Type>(count + (stops ? 1 : 0), word), step,
      body.getArguments(), *stepDeterminism)};
  llvm::SmallVector<mlir::Value> yielded;
  mlir::ValueRange accumulators{apply.getResults()};
  if (stops) {
    const semantics::Type *boolType{signature->arguments[1]->arguments[1]};
    mlir::Value more{
        Coerce(apply.getResult(0), Represent(boolType)->type, loc)};
    yielded.push_back(merk::HasFunctorOp::create(builder_, loc, more,
        merk::FunctorAttr::get(&context_, "yes", 0)));
    accumulators = accumulators.drop_front();
  } else {
    yielded.push_back(mlir::arith::ConstantIntOp::create(
        builder_, loc, builder_.getI1Type(), 1));
  }
  yielded.append(accumulators.begin(), accumulators.end());
  merk::YieldOp::create(builder_, loc, yielded);
  builder_.setInsertionPointAfter(aggregate);
  merk::SucceedOp::create(builder_, loc, aggregate.getResults());
}

void MerkGenerator::GenerateBody(const ProcedureBody &body, bool words) {
  const Procedure &procedure{*body.procedure};
  parser::CharBlock at{procedure.symbol->source};
  failed_ = false;
  inBody_ = true;
  notYet_.clear();
  llvm::scope_exit leave{[&] { inBody_ = false; }};
  const auto *polymorphism{polymorphism_.lookup(&body)};
  if (polymorphism &&
      polymorphism->unchecked != semantics::PolymorphismUnchecked::None) {
    NotYet(at, "this procedure's type_infos");
    return;
  }
  // A clause of an existentially typed procedure has the type it chose where
  // the declaration, and so its callers, have a type variable.
  std::size_t hiddenOutputs{
      body.method ? std::size_t{0} : HiddenOutputs(procedure).size()};
  llvm::SmallVector<const semantics::Type *> types;
  if (hiddenOutputs != 0) {
    llvm::ArrayRef<const semantics::Type *> declared{
        types_.SignatureOf(*procedure.symbol)->arguments};
    types.append(declared.begin(), declared.end());
  } else {
    for (BodyVar var : body.head) {
      types.push_back(body.vars[var].type);
    }
  }
  merk::ProcOp proc{body.method ? MethodProc(body)
          : words              ? WordProcFor(body)
                               : ProcFor(procedure, types, at)};
  if (!proc) {
    return;
  }
  if (procedure.symbol->exported && !body.method && !words) {
    proc.setPublic();
  }
  values_.clear();
  typeInfos_.clear();
  dictionaries_.clear();
  givenVars_.clear();
  givenTypes_.clear();
  givenInfos_.clear();
  current_ = polymorphism;
  procName_ = proc.getSymName().str();
  wordOutputs_.reset();
  mlir::Block *entry{proc.addEntryBlock()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(entry);
  llvm::SmallVector<BodyVar> outputs;
  unsigned input{0};
  if (polymorphism) {
    for (const semantics::HiddenArgument &argument : polymorphism->hidden) {
      if (argument.output) {
        continue;
      }
      if (argument.typeInfo) {
        typeInfos_[argument.typeInfo->parameter] = entry->getArgument(input++);
      } else {
        dictionaries_[argument.dictionary->parameter] =
            entry->getArgument(input++);
      }
    }
    // A constrained type variable's is in a typeclass_info.
    for (auto [var, source] : llvm::enumerate(polymorphism->sources)) {
      if (source && !typeInfos_.lookup(var)) {
        mlir::Value value{TypeInfoValue(*source, Location(at), at)};
        if (!value) {
          failed_ = true;
          break;
        }
        typeInfos_[var] = value;
      }
    }
  }
  llvm::SmallVector<BodyVar> wordOutputs;
  for (auto [var, mode, type] : llvm::zip(body.head, procedure.modes, types)) {
    if (words && IsOutput(mode)) {
      wordOutputs.push_back(var);
    }
    if (Represent(type)->dummy || IsUnused(mode)) {
      continue;
    }
    if (IsOutput(mode)) {
      outputs.push_back(var);
    } else {
      values_[var] = entry->getArgument(input++);
      // Here, whatever the body makes of it.
      rootTypes_.insert(type);
      if (NoteRoot(values_[var], type)) {
        annotated_.insert(values_[var]);
      }
    }
  }
  if (words) {
    wordOutputs_ = wordOutputs;
  }
  Goal(*body.body, body);
  if (failed_) {
    // Calls already made of it keep its declaration.
    proc.getBody().dropAllReferences();
    proc.getBody().getBlocks().clear();
    if (!notYet_.empty()) {
      StubBody(proc, at);
      return;
    }
    proc.setPrivate();
    return;
  }
  if (Terminated()) {
    return; // it cannot succeed
  }
  llvm::SmallVector<mlir::Value> results;
  if (words) {
    results = WordResults(wordOutputs, body, Location(at));
    if (llvm::is_contained(results, mlir::Value{})) {
      failed_ = true;
      return;
    }
    merk::SucceedOp::create(builder_, Location(at), results);
    EraseUnwordedOutputs(proc.getBody());
    return;
  }
  if (hiddenOutputs != 0) {
    // The clauses give the same types, but a clause that cannot succeed here
    // may have had its goals in a branch left behind.
    const semantics::PassedInfos *given{nullptr};
    for (const semantics::PassedInfos &clause : polymorphism->results) {
      if (llvm::all_of(clause.typeInfos,
              [&](const semantics::TypeInfo *i) { return CanMake(*i); }) &&
          llvm::all_of(clause.dictionaries,
              [&](const semantics::Dictionary *d) { return CanMake(*d); })) {
        given = &clause;
        break;
      }
    }
    if (!given) {
      NotYet(at, "this procedure's existential types");
      proc.getBody().dropAllReferences();
      proc.getBody().getBlocks().clear();
      proc.setPrivate();
      return;
    }
    for (const semantics::TypeInfo *info : given->typeInfos) {
      results.push_back(TypeInfoValue(*info, Location(at), at));
    }
    for (const semantics::Dictionary *dictionary : given->dictionaries) {
      results.push_back(DictionaryValue(*dictionary, Location(at), at));
    }
  }
  for (BodyVar var : outputs) {
    mlir::Type declared{
        proc.getFunctionType().getResult(static_cast<unsigned>(results.size()))};
    // A clause's type with no values, at a type variable: a word that nothing
    // reads.
    if (Represent(body.vars[var].type)->dummy) {
      results.push_back(merk::ToWordOp::create(builder_, Location(at), declared,
          IntegerConstant(builder_.getI64Type(), 0, Location(at))));
    } else {
      results.push_back(Coerce(ValueOf(var), declared, Location(at)));
    }
  }
  if (llvm::is_contained(results, mlir::Value{})) {
    failed_ = true;
    return;
  }
  merk::SucceedOp::create(builder_, Location(at), results);
}

mlir::OwningOpRef<mlir::ModuleOp> MerkGenerator::Generate() {
  context_.loadDialect<merk::MerkDialect, mlir::arith::ArithDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> owned{
      mlir::ModuleOp::create(builder_.getUnknownLoc(), moduleName_)};
  module_ = *owned;
  target_.Record(module_);
  bool interface{false};
  for (const parser::Item &item : tree_.items) {
    if (const auto *section{std::get_if<const parser::SectionDecl *>(&item)}) {
      interface = (*section)->section == parser::Section::Interface;
    } else if (const auto *decl{
                   std::get_if<const parser::AbstractTypeDecl *>(&item)};
        decl && interface) {
      for (const semantics::Symbol *symbol :
          resolution_.module().Lookup((*decl)->name->name)) {
        if (symbol->kind == semantics::SymbolKind::Type &&
            symbol->arity == (*decl)->parameters.size()) {
          abstractTypes_.insert(symbol);
        }
      }
    }
  }
  // A procedure with a foreign_proc for C runs its C code instead of its
  // clauses.
  for (const Procedure &procedure : modes_.procedures()) {
    if (const parser::ForeignProcPragma *pragma{
            modes_.ForeignProcOf(procedure)}) {
      GenerateForeign(procedure, *pragma);
    }
  }
  for (const ProcedureBody &body : determinism_.bodies()) {
    if (!body.method && body.procedure->symbol->module == moduleName_ &&
        !modes_.ForeignProcOf(*body.procedure)) {
      bodies_[body.procedure] = &body;
    }
  }
  for (const ProcedureBody &body : determinism_.bodies()) {
    if ((body.procedure->symbol->module == moduleName_ || body.method) &&
        (body.method || !modes_.ForeignProcOf(*body.procedure))) {
      GenerateBody(body);
    }
  }
  for (const Procedure &procedure : modes_.procedures()) {
    if (procedure.symbol->module == moduleName_ &&
        !modes_.ForeignProcOf(procedure) &&
        modes_.UncheckedReason(procedure) != semantics::Unchecked::None) {
      GenerateUnchecked(procedure);
    }
  }
  for (const Procedure &procedure : modes_.procedures()) {
    if (procedure.symbol->module == moduleName_ &&
        IsAggregate(*procedure.symbol) && !bodies_.contains(&procedure) &&
        !modes_.ForeignProcOf(procedure)) {
      GenerateAggregate(procedure);
    }
  }
  // The word copies that bodies called, and those that copies call; one of a
  // procedure whose body failed has no body either.
  for (std::size_t i{0}; i < pendingCopies_.size(); ++i) {
    const ProcedureBody &body{*pendingCopies_[i]};
    merk::ProcOp typed{procs_.lookup(body.procedure)};
    if (typed && !typed.isExternal()) {
      GenerateBody(body, /*words=*/true);
    }
  }
  GenerateInstances();
  // An exported type's descriptor is here, for the modules that import it,
  // which may not see its constructors. One that cannot be made yet is left
  // to those that use it, which fail to link.
  for (const semantics::Symbol *symbol : resolution_.module().symbols()) {
    if (symbol->kind != semantics::SymbolKind::Type || !symbol->exported ||
        !std::holds_alternative<const parser::DuTypeDefn *>(
            symbol->declaration)) {
      continue;
    }
    semantics::TypeContext &context{types_.types()};
    llvm::SmallVector<const semantics::Type *> variables;
    for (unsigned i{0}; i < symbol->arity; ++i) {
      variables.push_back(context.Variable(i));
    }
    const semantics::Type *canonical{context.Named(*symbol, variables)};
    llvm::StringSet<> seen;
    std::optional<Represented> represented{Represent(canonical)};
    if (represented && (represented->dummy ||
                           Comparable(represented->type, seen))) {
      EnsureDescriptor(*canonical, symbol->source);
    }
  }
  // Last, as each brings the procedures that test and compare its type.
  for (const semantics::Type *type : rootTypes_) {
    EnsureRootDescriptors(type);
  }
  if (!foreignCode_.empty()) {
    // The module's own foreign_decl and foreign_code for C come first.
    std::string code{kForeignPrelude};
    for (const parser::Item &item : tree_.items) {
      const parser::ForeignCode *foreign{nullptr};
      if (const auto *decl{std::get_if<const parser::ForeignDeclPragma *>(&item)};
          decl && (*decl)->language == parser::ForeignLanguage::C) {
        foreign = (*decl)->code;
      } else if (const auto *c{
                     std::get_if<const parser::ForeignCodePragma *>(&item)};
          c && (*c)->language == parser::ForeignLanguage::C) {
        foreign = (*c)->code;
      }
      if (!foreign) {
        continue;
      }
      if (foreign->includeFile) {
        code += "#include \"" + foreign->text.str() + "\"\n";
      } else {
        code += LineDirective(foreign->tokens) + foreign->text.str() + "\n";
      }
    }
    code += foreignCode_;
    module_->setAttr(kForeignCodeAttrName, builder_.getStringAttr(code));
  }
  RegisterMessageTypes();
  // The program's entry: the main module's main(di, uo).
  for (const ProcedureBody &body : determinism_.bodies()) {
    const semantics::Symbol &symbol{*body.procedure->symbol};
    if (!body.method && symbol.module == moduleName_ &&
        symbol.kind == semantics::SymbolKind::Predicate &&
        symbol.name == "main" && symbol.arity == 2) {
      if (merk::ProcOp proc{procs_.lookup(body.procedure)}) {
        builder_.setInsertionPointToEnd(module_.getBody());
        merk::EntryOp::create(
            builder_, Location(symbol.source), proc.getSymName());
      }
    }
  }
  // A goal's solutions give every variable it binds; only those used after
  // it stay.
  PruneSolutions(*owned);
  return owned;
}

} // namespace mlirgen

mlir::OwningOpRef<mlir::ModuleOp> GenerateMerk(mlir::MLIRContext &context,
    llvm::StringRef module, const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, const parser::AllSources &sources,
    const semantics::NameResolution &resolution, semantics::TypeChecking &types,
    semantics::ModeChecking &modes,
    const semantics::DeterminismChecking &determinism,
    const semantics::PolymorphismAnalysis &polymorphism,
    const semantics::LoopTranslation &loops, const TargetInfo &target,
    parser::Messages &messages, const GenerationOptions &options) {
  return mlirgen::MerkGenerator{context, module, tree, tokens, sources, resolution, types, modes,
      determinism, polymorphism, loops, target, messages, options}
      .Generate();
}

} // namespace merkurius::codegen
