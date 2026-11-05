//===-- runtime/write.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// io.write and io.print: a walker over a value by its type_info's descriptor
// (design document, "Type descriptors"), which writes it as mmc's programs do.
// The operators are the parser's, whose priorities bind more tightly the higher
// they are: here, as in the manual and in what a term's context allows, the
// lower.
//
//===----------------------------------------------------------------------===//

#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

extern "C" void merk_write_float(double number);

namespace {

using Word = std::uint64_t;

// The fields' order and types are lib/CodeGen/Lowering.cpp's LowerTypeCtors.
struct Functor {
  const char *name;
  std::uint32_t arity;
  std::uint8_t tag;
  std::int32_t secondary;
  std::int64_t constant;
  const Word *fields;
};

struct TypeCtor {
  std::uint8_t kind;
  std::uint8_t flags;
  std::uint16_t arity;
  std::uint32_t functorCount;
  const char *module;
  const char *name;
  const void *unify;
  const void *compare;
  const Functor *functors;
  std::uint16_t byTag[8];
  const std::uint16_t *constants;
  const std::uint16_t *shared;
};

// The order of Lowering.cpp's DescriptorKind.
enum Kind : std::uint8_t {
  Int,
  Uint,
  Int8,
  Int16,
  Int32,
  Int64,
  Uint8,
  Uint16,
  Uint32,
  Uint64,
  Float,
  Char,
  String,
  Array,
  Dummy,
  Void,
  Enum,
  NoTag,
  Cells,
  Tuple,
};

// byTag's marks, for a tag whose word is a constant, and for one that
// constructors share.
constexpr std::uint16_t kConstantTag{0xffff};
constexpr std::uint16_t kSharedTag{0xfffe};

constexpr Word kTagMask{7};
constexpr unsigned kTagBits{3};
// A field's type: a type_info, or a type parameter of its term's type.
constexpr Word kVariableTag{2};

enum Specifier { FX, FY, FXY, XFX, XFY, YFX };

struct Operator {
  std::string_view name;
  Specifier specifier;
  int priority;
};

constexpr int kOperatorCount{
#define MERK_OPERATOR(name, specifier, priority) +1
#include "../lib/Parser/operators.def"
#undef MERK_OPERATOR
};

// By name, for the lookups of a term's every functor.
constexpr std::array<Operator, kOperatorCount> SortedOperators() {
  std::array<Operator, kOperatorCount> table{{
#define MERK_OPERATOR(name, specifier, priority) {name, specifier, 1500 - priority},
#include "../lib/Parser/operators.def"
#undef MERK_OPERATOR
  }};
  for (int i{1}; i < kOperatorCount; ++i) {
    Operator item{table[i]};
    int j{i};
    for (; j > 0 && item.name < table[j - 1].name; --j) {
      table[j] = table[j - 1];
    }
    table[j] = item;
  }
  return table;
}

constexpr std::array<Operator, kOperatorCount> kOperators{SortedOperators()};

// A term's context lets it have any priority up to this, without parentheses.
constexpr int kAnyPriority{1200};
constexpr int kArgumentPriority{999};

const Operator *FirstOperator(std::string_view name) {
  int low{0}, high{kOperatorCount};
  while (low < high) {
    int middle{(low + high) / 2};
    if (kOperators[middle].name < name) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low < kOperatorCount && kOperators[low].name == name ? &kOperators[low]
                                                              : nullptr;
}

// The form of an operator with these operands, if it has one.
const Operator *FindOperator(std::string_view name, unsigned arity) {
  const Operator *op{FirstOperator(name)};
  for (; op && op != kOperators.data() + kOperatorCount && op->name == name; ++op) {
    bool prefix{op->specifier == FX || op->specifier == FY};
    if (arity == 1 ? prefix : arity == 2 && !prefix) {
      return op;
    }
  }
  return nullptr;
}

void Put(std::string_view text) { std::fwrite(text.data(), 1, text.size(), stdout); }

// A code point, or a byte that is not part of one, as a quoted name's, a
// character's or a string's.
void PutEscaped(std::uint32_t c, const char *bytes, std::size_t size) {
  switch (c) {
  case '\\':
    return Put("\\\\");
  case '\'':
    return Put("\\'");
  case '"':
    return Put("\\\"");
  case '\a':
    return Put("\\a");
  case '\b':
    return Put("\\b");
  case '\f':
    return Put("\\f");
  case '\n':
    return Put("\\n");
  case '\r':
    return Put("\\r");
  case '\t':
    return Put("\\t");
  case '\v':
    return Put("\\v");
  default:
    break;
  }
  if (c < 0x20 || (c >= 0x7f && c <= 0x9f)) {
    std::printf("\\%03" PRIo32 "\\", c);
  } else {
    Put({bytes, size});
  }
}

// The length of the UTF-8 sequence at `s`, and its code point; 1 for a byte
// that begins none.
std::size_t Decode(const unsigned char *s, std::uint32_t &c) {
  std::size_t size{*s < 0x80 ? 1u
          : (*s >> 5) == 6   ? 2u
          : (*s >> 4) == 14  ? 3u
          : (*s >> 3) == 30  ? 4u
                             : 0u};
  if (size == 0) {
    c = 0xfffd;
    return 1;
  }
  c = size == 1 ? *s : *s & (0xff >> (size + 1));
  for (std::size_t i{1}; i < size; ++i) {
    if ((s[i] >> 6) != 2) {
      c = 0xfffd;
      return 1;
    }
    c = (c << 6) | (s[i] & 0x3f);
  }
  return size;
}

void PutEscapedText(const char *text) {
  const auto *s{reinterpret_cast<const unsigned char *>(text)};
  while (*s) {
    std::uint32_t c;
    std::size_t size{Decode(s, c)};
    PutEscaped(c, reinterpret_cast<const char *>(s), size);
    s += size;
  }
}

void PutQuoted(const char *name) {
  std::string_view text{name};
  bool plain{text == "[]" || text == "{}" || text == "!"};
  if (!plain && !text.empty() && text[0] >= 'a' && text[0] <= 'z') {
    plain = true;
    for (char c : text) {
      plain = plain && ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                           (c >= '0' && c <= '9') || c == '_');
    }
  }
  if (plain) {
    Put(text);
    return;
  }
  std::putc('\'', stdout);
  PutEscapedText(name);
  std::putc('\'', stdout);
}

void PutChar(std::uint32_t c, bool quoted) {
  char bytes[4];
  std::size_t size;
  if (c < 0x80) {
    bytes[0] = static_cast<char>(c);
    size = 1;
  } else if (c < 0x800) {
    bytes[0] = static_cast<char>(0xc0 | (c >> 6));
    bytes[1] = static_cast<char>(0x80 | (c & 0x3f));
    size = 2;
  } else if (c < 0x10000) {
    bytes[0] = static_cast<char>(0xe0 | (c >> 12));
    bytes[1] = static_cast<char>(0x80 | ((c >> 6) & 0x3f));
    bytes[2] = static_cast<char>(0x80 | (c & 0x3f));
    size = 3;
  } else {
    bytes[0] = static_cast<char>(0xf0 | (c >> 18));
    bytes[1] = static_cast<char>(0x80 | ((c >> 12) & 0x3f));
    bytes[2] = static_cast<char>(0x80 | ((c >> 6) & 0x3f));
    bytes[3] = static_cast<char>(0x80 | (c & 0x3f));
    size = 4;
  }
  if (!quoted) {
    Put({bytes, size});
    return;
  }
  std::putc('\'', stdout);
  PutEscaped(c, bytes, size);
  std::putc('\'', stdout);
}

// What a type parameter stands for, in the type whose arguments these are.
struct Environment {
  const Word *arguments;
  const Environment *parent;
};

// A type: its constructor, and its arguments, in the environment they were
// made in. A walker does not allocate, so a field's type stays as it is.
struct Type {
  const TypeCtor *ctor;
  Environment environment;
};

Type Resolve(Word type, const Environment *environment) {
  while ((type & kTagMask) == kVariableTag) {
    type = environment->arguments[type >> kTagBits];
    environment = environment->parent;
  }
  if ((type & kTagMask) == 0) {
    return {reinterpret_cast<const TypeCtor *>(type), {nullptr, environment}};
  }
  const Word *cell{reinterpret_cast<const Word *>(type - 1)};
  const auto *ctor{reinterpret_cast<const TypeCtor *>(cell[0])};
  // A tuple's cell has the number of its arguments in word 1.
  return {ctor, {cell + (ctor->kind == Tuple ? 2 : 1), environment}};
}

// Whether a field's type is the type of the term it is in, as a list's tail
// is, so that a loop can go on with that type.
bool IsSameType(Word field, const Type &type) {
  if ((field & kTagMask) == 0) {
    return reinterpret_cast<const TypeCtor *>(field) == type.ctor;
  }
  if ((field & kTagMask) != 1) {
    return false;
  }
  const Word *cell{reinterpret_cast<const Word *>(field - 1)};
  if (reinterpret_cast<const TypeCtor *>(cell[0]) != type.ctor ||
      type.ctor->kind == Tuple) {
    return false;
  }
  for (unsigned i{0}; i < type.ctor->arity; ++i) {
    if (cell[1 + i] != ((Word{i} << kTagBits) | kVariableTag)) {
      return false;
    }
  }
  return true;
}

void WriteValue(Word value, Word type, const Environment *environment, int context);
void WriteTyped(Word value, const Type &type, int context);

// The functor of a term, and where its fields are.
const Functor *FunctorOf(Word &value, const TypeCtor *ctor, const Word *&fields) {
  switch (ctor->kind) {
  case Enum:
    return &ctor->functors[value];
  case Dummy:
    return &ctor->functors[0];
  case NoTag:
    fields = &value;
    return &ctor->functors[0];
  case Tuple: {
    const Functor *functor{&ctor->functors[0]};
    // A tuple of one argument is that argument, as a NoTag is.
    fields = functor->arity == 1
        ? &value
        : reinterpret_cast<const Word *>(value - functor->tag);
    return functor;
  }
  default:
    break;
  }
  unsigned tag{static_cast<unsigned>(value & kTagMask)};
  std::uint16_t index{ctor->byTag[tag]};
  if (index == kConstantTag) {
    return &ctor->functors[ctor->constants[value >> kTagBits]];
  }
  const Word *cell{reinterpret_cast<const Word *>(value - tag)};
  if (index == kSharedTag) {
    fields = cell + 1;
    return &ctor->functors[ctor->shared[cell[0]]];
  }
  fields = cell;
  return &ctor->functors[index];
}

void WriteList(const Functor *functor, const Word *fields, const Type &type) {
  std::putc('[', stdout);
  for (;;) {
    WriteValue(fields[0], functor->fields[0], &type.environment, kArgumentPriority);
    Word tail{fields[1]};
    Word tailType{functor->fields[1]};
    if (IsSameType(tailType, type)) {
      const Word *tailFields{nullptr};
      const Functor *next{FunctorOf(tail, type.ctor, tailFields)};
      std::string_view name{next->name};
      if (name == "[|]" && next->arity == 2) {
        Put(", ");
        functor = next;
        fields = tailFields;
        continue;
      }
      if (name == "[]" && next->arity == 0) {
        break;
      }
    }
    Put(" | ");
    WriteValue(tail, tailType, &type.environment, kAnyPriority);
    break;
  }
  std::putc(']', stdout);
}

void WriteTerm(Word term, const Type &type, int context) {
  const Word *fields{nullptr};
  const Functor *functor{FunctorOf(term, type.ctor, fields)};
  std::string_view name{functor->name};
  unsigned arity{functor->arity};
  const Environment *environment{&type.environment};
  auto field{[&](unsigned i, int priority) {
    WriteValue(fields[i], functor->fields[i], environment, priority);
  }};
  if (arity == 0) {
    if (context < kAnyPriority && FirstOperator(name)) {
      std::putc('(', stdout);
      Put(name);
      std::putc(')', stdout);
    } else {
      PutQuoted(functor->name);
    }
    return;
  }
  if (name == "[|]" && arity == 2) {
    WriteList(functor, fields, type);
    return;
  }
  if (name == "{}") {
    if (arity == 1) {
      Put("{ ");
      field(0, kAnyPriority);
      Put(" }");
      return;
    }
    std::putc('{', stdout);
    for (unsigned i{0}; i < arity; ++i) {
      if (i != 0) {
        Put(", ");
      }
      field(i, kArgumentPriority);
    }
    std::putc('}', stdout);
    return;
  }
  if (const Operator *op{FindOperator(name, arity)}) {
    int priority{op->priority};
    bool parentheses{priority > context};
    if (parentheses) {
      std::putc('(', stdout);
    }
    if (arity == 1) {
      Put(name);
      std::putc(' ', stdout);
      field(0, op->specifier == FY ? priority : priority - 1);
    } else if (op->specifier == FXY) {
      Put(name);
      std::putc(' ', stdout);
      field(0, priority - 1);
      std::putc(' ', stdout);
      field(1, priority);
    } else {
      field(0, op->specifier == YFX ? priority : priority - 1);
      if (name == ",") {
        Put(", ");
      } else {
        std::putc(' ', stdout);
        Put(name);
        std::putc(' ', stdout);
      }
      field(1, op->specifier == XFY ? priority : priority - 1);
    }
    if (parentheses) {
      std::putc(')', stdout);
    }
    return;
  }
  PutQuoted(functor->name);
  std::putc('(', stdout);
  for (unsigned i{0}; i < arity; ++i) {
    if (i != 0) {
      Put(", ");
    }
    field(i, kArgumentPriority);
  }
  std::putc(')', stdout);
}

void WriteInteger(Word value, unsigned bits, bool isSigned, const char *suffix) {
  if (isSigned) {
    std::int64_t number{static_cast<std::int64_t>(value << (64 - bits)) >>
        (64 - bits)};
    std::printf("%" PRId64 "%s", number, suffix);
  } else {
    std::printf("%" PRIu64 "%s", value, suffix);
  }
}

void WriteTyped(Word value, const Type &type, int context) {
  switch (type.ctor->kind) {
  case Int:
    return WriteInteger(value, 64, true, "");
  case Uint:
    return WriteInteger(value, 64, false, "u");
  case Int8:
    return WriteInteger(value, 8, true, "i8");
  case Int16:
    return WriteInteger(value, 16, true, "i16");
  case Int32:
    return WriteInteger(value, 32, true, "i32");
  case Int64:
    return WriteInteger(value, 64, true, "i64");
  case Uint8:
    return WriteInteger(value, 64, false, "u8");
  case Uint16:
    return WriteInteger(value, 64, false, "u16");
  case Uint32:
    return WriteInteger(value, 64, false, "u32");
  case Uint64:
    return WriteInteger(value, 64, false, "u64");
  case Float: {
    double number;
    std::memcpy(&number, &value, sizeof number);
    merk_write_float(number);
    return;
  }
  case Char:
    return PutChar(static_cast<std::uint32_t>(value), true);
  case String:
    std::putc('"', stdout);
    PutEscapedText(reinterpret_cast<const char *>(value));
    std::putc('"', stdout);
    return;
  case Array: {
    const Word *array{reinterpret_cast<const Word *>(value)};
    Word element{(Word{0} << kTagBits) | kVariableTag};
    Put("array([");
    for (Word i{0}; i < array[0]; ++i) {
      if (i != 0) {
        Put(", ");
      }
      WriteValue(array[1 + i], element, &type.environment, kArgumentPriority);
    }
    Put("])");
    return;
  }
  case Void:
    // Nothing has a value of this type.
    std::abort();
  default:
    return WriteTerm(value, type, context);
  }
}

void WriteValue(Word value, Word type, const Environment *environment, int context) {
  Type resolved{Resolve(type, environment)};
  WriteTyped(value, resolved, context);
}

// merk_write's flags, as io.m passes them.
constexpr Word kPrint{1};
constexpr Word kLine{2};

} // namespace

// The shortest of 15 to 17 digits that reads back as the same float, and
// always a sign of being one.
extern "C" void merk_write_float(double number) {
  char buffer[64];
  if (std::isnan(number)) {
    std::strcpy(buffer, "nan");
  } else if (std::isinf(number)) {
    std::strcpy(buffer, number < 0 ? "-infinity" : "infinity");
  } else {
    for (int precision{15}; precision <= 17; ++precision) {
      std::snprintf(buffer, sizeof buffer, "%.*g", precision, number);
      if (std::strtod(buffer, nullptr) == number) {
        break;
      }
    }
    if (std::strpbrk(buffer, ".e") == nullptr) {
      std::strcat(buffer, ".0");
    }
  }
  std::fputs(buffer, stdout);
}

// io.write and io.print, whose print writes a string or a character itself,
// and their _line forms.
extern "C" void merk_write(Word typeInfo, Word value, Word flags) {
  Type type{Resolve(typeInfo, nullptr)};
  if ((flags & kPrint) != 0 && type.ctor->kind == String) {
    std::fputs(reinterpret_cast<const char *>(value), stdout);
  } else if ((flags & kPrint) != 0 && type.ctor->kind == Char) {
    PutChar(static_cast<std::uint32_t>(value), false);
  } else {
    WriteTyped(value, type, kAnyPriority);
  }
  if ((flags & kLine) != 0) {
    std::putc('\n', stdout);
  }
}
