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
// An open term (design document, Runtime system, "Support for open terms")
// is written in its open layout, as from_open would give it, with each
// unbound variable as _1, _2 and so on, numbered in the order the write
// meets them, the same variable with the same number. A cell already on
// the path from the term's root, as a cyclic term has, is written as ...,
// so that the write ends.
//
//===----------------------------------------------------------------------===//

#include "array.h"
#include "descriptor.h"
#include "heap.h"
#include "open.h"
#include "process.h"
#include "record.h"
#include "walk.h"
#include "word-map.h"
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

extern "C" void merk_format_float(double number, char *buffer);
extern "C" void merk_format_float32(float number, char *buffer);
extern "C" char *merk_string_new(std::int64_t bytes);

namespace {

using namespace merkurius::runtime;

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

// Where the writer writes: a stream, or a buffer that becomes a string, for
// string.string. It has no destructor, which would need the C++ runtime's
// unwinding, which the runtime does not link: Take frees the buffer.
class Out {
public:
  explicit Out(std::FILE *file) : file_{file} {}
  Out() = default;
  Out(const Out &) = delete;
  Out &operator=(const Out &) = delete;

  void Put(std::string_view text) {
    if (file_) {
      std::fwrite(text.data(), 1, text.size(), file_);
      return;
    }
    if (size_ + text.size() > capacity_) {
      capacity_ = capacity_ * 2 > size_ + text.size() ? capacity_ * 2
                                                      : size_ + text.size();
      data_ = static_cast<char *>(std::realloc(data_, capacity_));
      if (!data_) {
        std::abort();
      }
    }
    std::memcpy(data_ + size_, text.data(), text.size());
    size_ += text.size();
  }
  void Put(char c) { Put(std::string_view{&c, 1}); }
  // For short text only, an integer and its suffix.
  __attribute__((format(printf, 2, 3))) void Printf(const char *format, ...) {
    char buffer[64];
    std::va_list arguments;
    va_start(arguments, format);
    int size{std::vsnprintf(buffer, sizeof buffer, format, arguments)};
    va_end(arguments);
    Put(std::string_view{buffer, static_cast<std::size_t>(size)});
  }

  // What was written to the buffer, as a string on the heap.
  char *Take() {
    char *text{merk_string_new(static_cast<std::int64_t>(size_))};
    if (size_ != 0) {
      std::memcpy(text, data_, size_);
    }
    std::free(data_);
    data_ = nullptr;
    size_ = capacity_ = 0;
    return text;
  }

private:
  std::FILE *file_{nullptr};
  char *data_{nullptr};
  std::size_t size_{0};
  std::size_t capacity_{0};
};


// A code point, or a byte that is not part of one, as a quoted name's, a
// character's or a string's.
void PutEscaped(Out &out, std::uint32_t c, const char *bytes, std::size_t size) {
  switch (c) {
  case '\\':
    return out.Put("\\\\");
  case '\'':
    return out.Put("\\'");
  case '"':
    return out.Put("\\\"");
  case '\a':
    return out.Put("\\a");
  case '\b':
    return out.Put("\\b");
  case '\f':
    return out.Put("\\f");
  case '\n':
    return out.Put("\\n");
  case '\r':
    return out.Put("\\r");
  case '\t':
    return out.Put("\\t");
  case '\v':
    return out.Put("\\v");
  default:
    break;
  }
  if (c < 0x20 || (c >= 0x7f && c <= 0x9f)) {
    out.Printf("\\%03" PRIo32 "\\", c);
  } else {
    out.Put({bytes, size});
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

// A byte that is not part of a well-formed sequence is written as U+FFFD, as
// mmc does, so that what is written is UTF-8.
void PutEscapedText(Out &out, const char *text) {
  const auto *s{reinterpret_cast<const unsigned char *>(text)};
  while (*s) {
    std::uint32_t c;
    std::size_t size{Decode(s, c)};
    if (size == 1 && *s >= 0x80) {
      out.Put("\xEF\xBF\xBD");
    } else {
      PutEscaped(out, c, reinterpret_cast<const char *>(s), size);
    }
    s += size;
  }
}

void PutQuoted(Out &out, const char *name) {
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
    out.Put(text);
    return;
  }
  out.Put('\'');
  PutEscapedText(out, name);
  out.Put('\'');
}

void PutChar(Out &out, std::uint32_t c, bool quoted) {
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
    out.Put({bytes, size});
    return;
  }
  out.Put('\'');
  PutEscaped(out, c, bytes, size);
  out.Put('\'');
}

// A write of an open term: its variables' numbers, and the cells on its
// path, kept in `path` and, in order, in `stack`.
struct OpenWrite {
  WordMap variables;
  Word next{1};
  WordMap path;
  Word *stack{nullptr};
  std::size_t depth{0};
  std::size_t capacity{0};

  // Whether the cell was not on the path, which it is now.
  bool Enter(Word cell) {
    bool inserted;
    path.Insert(cell, 1, inserted);
    if (!inserted) {
      return false;
    }
    if (depth == capacity) {
      capacity = capacity ? 2 * capacity : 64;
      stack = static_cast<Word *>(std::realloc(stack, capacity * sizeof(Word)));
      if (!stack) {
        std::abort();
      }
    }
    stack[depth++] = cell;
    return true;
  }
  // The cells entered after the path was `to` deep leave it.
  void LeaveTo(std::size_t to) {
    while (depth > to) {
      path.Erase(stack[--depth]);
    }
  }
  void Free() {
    variables.Free();
    path.Free();
    std::free(stack);
  }
};

// The open state of the io.write call under way (merk_write,
// merk_string_of): one numbering of variables for its every open(T) value.
thread_local OpenWrite *callWrite{nullptr};

// An open word with its chain of bound variables followed.
Word Deref(Word value) {
  while (IsVariableReference(value)) {
    Word next{*reinterpret_cast<const Word *>(value - kOpenVariableTag)};
    if (next == value) {
      break;
    }
    value = next;
  }
  return value;
}

void WriteVariable(Out &out, Word variable, OpenWrite &open) {
  bool inserted;
  Word *number{open.variables.Insert(variable, open.next, inserted)};
  if (inserted) {
    ++open.next;
  }
  out.Printf("_%" PRIu64, *number);
}

// The functor of an open word of a NoTag, Tuple or Cells type, which deref
// gave and is not a variable, as FunctorOf gives an ordinary one's.
const Functor *OpenFunctorOf(
    Word &value, const TypeCtor *ctor, const Word *&fields) {
  if (ctor->kind == NoTag ||
      (ctor->kind == Tuple && ctor->functors[0].arity == 1)) {
    fields = &value;
    return &ctor->functors[0];
  }
  if (IsOpenConstant(ctor, value)) {
    fields = nullptr;
    return &ctor->functors[ctor->constants[value >> kTagBits]];
  }
  unsigned tag{static_cast<unsigned>(value & kTagMask)};
  const Word *cell{reinterpret_cast<const Word *>(value - tag)};
  unsigned before;
  const Functor *functor{OpenCellFunctor(ctor, tag, cell, before)};
  if (!functor || functor->hidden != 0) {
    std::fflush(stdout);
    std::fputs("Merkurius runtime: an open term with a constructor it cannot "
               "have\n",
        stderr);
    std::abort();
  }
  fields = cell + before;
  return functor;
}

void WriteValue(Out &out, Word value, Word type, const Environment *environment,
    int context, OpenWrite *open = nullptr);

void *AllocateCell(Word bytes) { return Allocate(bytes); }

// Element `i` of an array of the element type, as a word: a record boxed,
// on the heap, which only grows it, as the runtime's other allocations do.
Word ElementWord(
    const Word *array, Word i, Word element, const Environment &environment) {
  if (!IsRecordArray(array)) {
    return LoadElement(array, i);
  }
  return BoxRecord(Resolve(element, &environment),
      ArrayElements(array) + array[1] * i, AllocateCell);
}
void WriteTyped(Out &out, Word value, const Type &type, int context,
    OpenWrite *open = nullptr);

// In an open term, a tail that is an unbound variable is written after |,
// and one whose cell is on the path as ....
void WriteList(Out &out, const Functor *functor, const Word *fields,
    const Type &type, OpenWrite *open) {
  out.Put('[');
  for (;;) {
    WriteValue(out, fields[0], functor->fields[0], &type.environment,
        kArgumentPriority, open);
    Word tail{open ? Deref(fields[1]) : fields[1]};
    Word tailType{functor->fields[1]};
    if (IsSameType(tailType, type) && !(open && IsVariableReference(tail))) {
      const Word *tailFields{nullptr};
      const Functor *next{open ? OpenFunctorOf(tail, type.ctor, tailFields)
                               : FunctorOf(tail, type.ctor, tailFields)};
      std::string_view name{next->name};
      if (name == "[|]" && next->arity == 2) {
        if (open && !open->Enter(tail)) {
          out.Put(" | ...");
          break;
        }
        out.Put(", ");
        functor = next;
        fields = tailFields;
        continue;
      }
      if (name == "[]" && next->arity == 0) {
        break;
      }
    }
    out.Put(" | ");
    WriteValue(out, tail, tailType, &type.environment, kAnyPriority, open);
    break;
  }
  out.Put(']');
}

void WriteTermFields(Out &out, Word term, const Functor *functor,
    const Word *fields, const Type &type, int context, OpenWrite *open);

// In an open term, a cell on the path is written as ..., and the cells the
// term enters leave the path after it.
void WriteTerm(Out &out, Word term, const Type &type, int context,
    OpenWrite *open = nullptr) {
  const Word *fields{nullptr};
  const Functor *functor{open ? OpenFunctorOf(term, type.ctor, fields)
                              : FunctorOf(term, type.ctor, fields)};
  if (!open || !fields || fields == &term) {
    WriteTermFields(out, term, functor, fields, type, context, open);
    return;
  }
  std::size_t depth{open->depth};
  if (!open->Enter(term)) {
    out.Put("...");
    return;
  }
  WriteTermFields(out, term, functor, fields, type, context, open);
  open->LeaveTo(depth);
}

void WriteTermFields(Out &out, Word term, const Functor *functor,
    const Word *fields, const Type &type, int context, OpenWrite *open) {
  std::string_view name{functor->name};
  unsigned arity{functor->arity};
  Environment existential{type.environment.arguments,
      type.environment.parent, fields - functor->hidden};
  const Environment *environment{
      functor->hidden != 0 ? &existential : &type.environment};
  if (arity == 0) {
    if (context < kAnyPriority && FirstOperator(name)) {
      out.Put('(');
      out.Put(name);
      out.Put(')');
    } else {
      PutQuoted(out, functor->name);
    }
    return;
  }
  if (name == "[|]" && arity == 2) {
    WriteList(out, functor, fields, type, open);
    return;
  }
  if (name == "{}") {
    if (arity == 1) {
      out.Put("{ ");
      WriteValue(
          out, fields[0], functor->fields[0], environment, kAnyPriority, open);
      out.Put(" }");
      return;
    }
    out.Put('{');
    for (unsigned i{0}; i < arity; ++i) {
      if (i != 0) {
        out.Put(", ");
      }
      WriteValue(out, fields[i], functor->fields[i], environment,
          kArgumentPriority, open);
    }
    out.Put('}');
    return;
  }
  if (const Operator *op{FindOperator(name, arity)}) {
    int priority{op->priority};
    bool parentheses{priority > context};
    if (parentheses) {
      out.Put('(');
    }
    if (arity == 1) {
      out.Put(name);
      out.Put(' ');
      WriteValue(out, fields[0], functor->fields[0], environment,
          op->specifier == FY ? priority : priority - 1, open);
    } else if (op->specifier == FXY) {
      out.Put(name);
      out.Put(' ');
      WriteValue(
          out, fields[0], functor->fields[0], environment, priority - 1, open);
      out.Put(' ');
      WriteValue(
          out, fields[1], functor->fields[1], environment, priority, open);
    } else {
      WriteValue(out, fields[0], functor->fields[0], environment,
          op->specifier == YFX ? priority : priority - 1, open);
      if (name == ",") {
        out.Put(", ");
      } else {
        out.Put(' ');
        out.Put(name);
        out.Put(' ');
      }
      WriteValue(out, fields[1], functor->fields[1], environment,
          op->specifier == XFY ? priority : priority - 1, open);
    }
    if (parentheses) {
      out.Put(')');
    }
    return;
  }
  PutQuoted(out, functor->name);
  out.Put('(');
  for (unsigned i{0}; i < arity; ++i) {
    if (i != 0) {
      out.Put(", ");
    }
    WriteValue(out, fields[i], functor->fields[i], environment,
        kArgumentPriority, open);
  }
  out.Put(')');
}

void WriteInteger(Out &out, Word value, unsigned bits, bool isSigned, const char *suffix) {
  if (isSigned) {
    std::int64_t number{static_cast<std::int64_t>(value << (64 - bits)) >>
        (64 - bits)};
    out.Printf("%" PRId64 "%s", number, suffix);
  } else {
    out.Printf("%" PRIu64 "%s", value, suffix);
  }
}

// An open word: a variable, or its value decoded from its type's open
// encoding (open.h).
void WriteOpen(
    Out &out, Word value, const Type &type, int context, OpenWrite &open) {
  const TypeCtor *ctor{type.ctor};
  // A notag's word is its argument's, so an unbound one is the constructor
  // around an unbound argument, which it can only be bound to.
  if ((ctor->kind == NoTag && (ctor->flags & kClosed) == 0) ||
      (ctor->kind == Tuple && ctor->functors[0].arity == 1)) {
    return WriteTerm(out, value, type, context, &open);
  }
  value = Deref(value);
  if (IsVariableReference(value)) {
    WriteVariable(out, value, open);
    return;
  }
  switch (ctor->kind) {
  case Int8:
  case Int16:
  case Int32:
  case Uint8:
  case Uint16:
  case Uint32:
  case Char:
  case Enum:
    // Shifted past the tag, and zero-extended, as WriteInteger takes it.
    return WriteTyped(out, value >> kTagBits, type, context);
  case Int:
  case Uint:
  case Int64:
  case Uint64:
  case Float:
  case Float32:
  case String:
    return WriteTyped(
        out, *reinterpret_cast<const Word *>(value), type, context);
  case Dummy:
  case Void:
    return WriteTyped(out, value, type, context);
  case Open:
    return WriteOpen(out, value,
        Resolve(type.environment.arguments[0], type.environment.parent),
        context, open);
  case NoTag:
    if ((ctor->flags & kClosed) != 0) {
      // A box of T's ordinary word, which closed(T)'s ordinary value is.
      return WriteTerm(
          out, *reinterpret_cast<const Word *>(value), type, context);
    }
    return WriteTerm(out, value, type, context, &open);
  case Tuple:
  case Cells:
    return WriteTerm(out, value, type, context, &open);
  default:
    std::fflush(stdout);
    std::fputs("Merkurius runtime: an open term of a type open terms cannot "
               "hold\n",
        stderr);
    std::abort();
  }
}

void WriteTyped(
    Out &out, Word value, const Type &type, int context, OpenWrite *open) {
  if (open) {
    return WriteOpen(out, value, type, context, *open);
  }
  switch (type.ctor->kind) {
  case Int:
    return WriteInteger(out, value, 64, true, "");
  case Uint:
    return WriteInteger(out, value, 64, false, "u");
  case Int8:
    return WriteInteger(out, value, 8, true, "i8");
  case Int16:
    return WriteInteger(out, value, 16, true, "i16");
  case Int32:
    return WriteInteger(out, value, 32, true, "i32");
  case Int64:
    return WriteInteger(out, value, 64, true, "i64");
  case Uint8:
    return WriteInteger(out, value, 64, false, "u8");
  case Uint16:
    return WriteInteger(out, value, 64, false, "u16");
  case Uint32:
    return WriteInteger(out, value, 64, false, "u32");
  case Uint64:
    return WriteInteger(out, value, 64, false, "u64");
  case Float: {
    double number;
    std::memcpy(&number, &value, sizeof number);
    char buffer[64];
    merk_format_float(number, buffer);
    out.Put(buffer);
    return;
  }
  case Float32: {
    // With its suffix, as a sized integer's, so that it reads back as a
    // float32; an infinity or a NaN has none, as it has no literal.
    float number;
    auto bits{static_cast<std::uint32_t>(value)};
    std::memcpy(&number, &bits, sizeof number);
    char buffer[64];
    merk_format_float32(number, buffer);
    out.Put(buffer);
    if (std::isfinite(number)) {
      out.Put("f32");
    }
    return;
  }
  case Char:
    return PutChar(out, static_cast<std::uint32_t>(value), true);
  case String:
    out.Put('"');
    PutEscapedText(out, reinterpret_cast<const char *>(value));
    out.Put('"');
    return;
  case Array: {
    const Word *array{reinterpret_cast<const Word *>(value)};
    Word element{(Word{0} << kTagBits) | kVariableTag};
    out.Put("array([");
    for (Word i{0}; i < ArraySize(array); ++i) {
      if (i != 0) {
        out.Put(", ");
      }
      WriteValue(out, ElementWord(array, i, element, type.environment), element,
          &type.environment, kArgumentPriority);
    }
    out.Put("])");
    return;
  }
  case Closure:
    // As mmc writes one.
    out.Put(type.ctor->name[0] == 'f' ? "'<<function>>'" : "'<<predicate>>'");
    return;
  case Void:
    // Nothing has a value of this type.
    std::abort();
  case Open:
    // T's open layout, numbered as the call's other open values are.
    return WriteOpen(out, value,
        Resolve(type.environment.arguments[0], type.environment.parent),
        context, *callWrite);
  default:
    return WriteTerm(out, value, type, context);
  }
}

void WriteValue(Out &out, Word value, Word type, const Environment *environment,
    int context, OpenWrite *open) {
  Type resolved{Resolve(type, environment)};
  WriteTyped(out, value, resolved, context, open);
}

// merk_write's flags, as io.m passes them.
constexpr Word kPrint{1};
constexpr Word kLine{2};

} // namespace

// The shortest of 15 to 17 digits that reads back as the same float, and
// always a sign of being one, as io.write_float and string.float_to_string
// give it, in a buffer of 64 bytes.
extern "C" void merk_format_float(double number, char *buffer) {
  if (std::isnan(number)) {
    std::strcpy(buffer, "nan");
  } else if (std::isinf(number)) {
    std::strcpy(buffer, number < 0 ? "-infinity" : "infinity");
  } else {
    for (int precision{15}; precision <= 17; ++precision) {
      std::snprintf(buffer, 64, "%.*g", precision, number);
      if (std::strtod(buffer, nullptr) == number) {
        break;
      }
    }
    if (std::strpbrk(buffer, ".e") == nullptr) {
      std::strcat(buffer, ".0");
    }
  }
}

extern "C" void merk_write_float(double number) {
  char buffer[64];
  merk_format_float(number, buffer);
  std::fputs(buffer, stdout);
}

// A float32 as merk_format_float gives a float: the shortest of 6 to 9
// digits that reads back as the same float32 (docs/INPROGRESS.md, Sized
// floating point types).
extern "C" void merk_format_float32(float number, char *buffer) {
  if (std::isnan(number)) {
    std::strcpy(buffer, "nan");
  } else if (std::isinf(number)) {
    std::strcpy(buffer, number < 0 ? "-infinity" : "infinity");
  } else {
    for (int precision{6}; precision <= 9; ++precision) {
      std::snprintf(buffer, 64, "%.*g", precision, static_cast<double>(number));
      if (std::strtof(buffer, nullptr) == number) {
        break;
      }
    }
    if (std::strpbrk(buffer, ".e") == nullptr) {
      std::strcat(buffer, ".0");
    }
  }
}

extern "C" void merk_write_float32(float number) {
  char buffer[64];
  merk_format_float32(number, buffer);
  std::fputs(buffer, stdout);
}

// io.write and io.print, whose print writes a string or a character itself,
// and their _line forms.
extern "C" void merk_write(Word typeInfo, Word value, Word flags) {
  Out out{stdout};
  OpenWrite write;
  callWrite = &write;
  Type type{Resolve(typeInfo, nullptr)};
  if ((flags & kPrint) != 0 && type.ctor->kind == String) {
    out.Put(reinterpret_cast<const char *>(value));
  } else if ((flags & kPrint) != 0 && type.ctor->kind == Char) {
    PutChar(out, static_cast<std::uint32_t>(value), false);
  } else {
    WriteTyped(out, value, type, kAnyPriority);
  }
  if ((flags & kLine) != 0) {
    out.Put('\n');
  }
  callWrite = nullptr;
  write.Free();
}

// What io.write writes, to a stream: the report of an uncaught exception
// (exception.cpp), which must not allocate on the process's heap.
void merkurius::runtime::WriteValue(std::FILE *file, Word typeInfo, Word value) {
  Out out{file};
  OpenWrite write;
  callWrite = &write;
  WriteTyped(out, value, Resolve(typeInfo, nullptr), kAnyPriority);
  callWrite = nullptr;
  write.Free();
}

// string.string: what io.write writes, as a string.
extern "C" char *merk_string_of(Word typeInfo, Word value) {
  Out out;
  OpenWrite write;
  callWrite = &write;
  WriteTyped(out, value, Resolve(typeInfo, nullptr), kAnyPriority);
  callWrite = nullptr;
  write.Free();
  return out.Take();
}

namespace {

void PutTypeName(Out &out, Word typeInfo);

// A type's arguments' names, separated by commas.
void PutTypeNames(Out &out, const Word *arguments, Word count) {
  for (Word i{0}; i < count; ++i) {
    if (i != 0) {
      out.Put(", ");
    }
    PutTypeName(out, arguments[i]);
  }
}

// A type's name as mmc's type_desc.type_name writes it: qualified by its
// module, but for a builtin type, a tuple's {A, B}, a function's
// func(A) = R, or ((func) = R) without arguments.
void PutTypeName(Out &out, Word typeInfo) {
  const TypeCtor *ctor;
  const Word *arguments{nullptr};
  Word arity{0};
  if ((typeInfo & kTagMask) == 0) {
    ctor = reinterpret_cast<const TypeCtor *>(typeInfo);
  } else {
    const Word *cell{reinterpret_cast<const Word *>(typeInfo - 1)};
    ctor = reinterpret_cast<const TypeCtor *>(cell[0]);
    bool variable{ctor->kind == Tuple || ctor->kind == Closure};
    arguments = cell + (variable ? 2 : 1);
    arity = variable ? cell[1] : ctor->arity;
  }
  // A descriptor of a builtin type, a tuple or a closure has no module.
  if (*ctor->module) {
    out.Put(ctor->module);
    out.Put('.');
  }
  bool function{ctor->kind == Closure && std::strcmp(ctor->name, "func") == 0};
  if (ctor->kind == Tuple) {
    out.Put('{');
    PutTypeNames(out, arguments, arity);
    out.Put('}');
  } else if (arity == 0) {
    out.Put(ctor->name);
  } else if (function && arity == 1) {
    out.Put("((func) = ");
    PutTypeName(out, arguments[0]);
    out.Put(')');
  } else if (function) {
    out.Put("func(");
    PutTypeNames(out, arguments, arity - 1);
    out.Put(") = ");
    PutTypeName(out, arguments[arity - 1]);
  } else {
    out.Put(ctor->name);
    out.Put('(');
    PutTypeNames(out, arguments, arity);
    out.Put(')');
  }
}

} // namespace

// The name of a type_info's type, as a string, for univ.det_univ_to_type's
// message.
extern "C" char *merk_type_name(Word typeInfo) {
  Out out;
  PutTypeName(out, typeInfo);
  return out.Take();
}
