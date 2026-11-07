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
#include "descriptor.h"
#include "walk.h"
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

extern "C" void merk_format_float(double number, char *buffer);
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

void WriteValue(Out &out, Word value, Word type, const Environment *environment, int context);
void WriteTyped(Out &out, Word value, const Type &type, int context);

void WriteList(Out &out, const Functor *functor, const Word *fields, const Type &type) {
  out.Put('[');
  for (;;) {
    WriteValue(out, fields[0], functor->fields[0], &type.environment, kArgumentPriority);
    Word tail{fields[1]};
    Word tailType{functor->fields[1]};
    if (IsSameType(tailType, type)) {
      const Word *tailFields{nullptr};
      const Functor *next{FunctorOf(tail, type.ctor, tailFields)};
      std::string_view name{next->name};
      if (name == "[|]" && next->arity == 2) {
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
    WriteValue(out, tail, tailType, &type.environment, kAnyPriority);
    break;
  }
  out.Put(']');
}

void WriteTerm(Out &out, Word term, const Type &type, int context) {
  const Word *fields{nullptr};
  const Functor *functor{FunctorOf(term, type.ctor, fields)};
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
    WriteList(out, functor, fields, type);
    return;
  }
  if (name == "{}") {
    if (arity == 1) {
      out.Put("{ ");
      WriteValue(out, fields[0], functor->fields[0], environment, kAnyPriority);
      out.Put(" }");
      return;
    }
    out.Put('{');
    for (unsigned i{0}; i < arity; ++i) {
      if (i != 0) {
        out.Put(", ");
      }
      WriteValue(out, fields[i], functor->fields[i], environment, kArgumentPriority);
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
          op->specifier == FY ? priority : priority - 1);
    } else if (op->specifier == FXY) {
      out.Put(name);
      out.Put(' ');
      WriteValue(out, fields[0], functor->fields[0], environment, priority - 1);
      out.Put(' ');
      WriteValue(out, fields[1], functor->fields[1], environment, priority);
    } else {
      WriteValue(out, fields[0], functor->fields[0], environment,
          op->specifier == YFX ? priority : priority - 1);
      if (name == ",") {
        out.Put(", ");
      } else {
        out.Put(' ');
        out.Put(name);
        out.Put(' ');
      }
      WriteValue(out, fields[1], functor->fields[1], environment,
          op->specifier == XFY ? priority : priority - 1);
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
    WriteValue(out, fields[i], functor->fields[i], environment, kArgumentPriority);
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

void WriteTyped(Out &out, Word value, const Type &type, int context) {
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
    for (Word i{0}; i < array[0]; ++i) {
      if (i != 0) {
        out.Put(", ");
      }
      WriteValue(out, array[1 + i], element, &type.environment, kArgumentPriority);
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
  default:
    return WriteTerm(out, value, type, context);
  }
}

void WriteValue(Out &out, Word value, Word type, const Environment *environment, int context) {
  Type resolved{Resolve(type, environment)};
  WriteTyped(out, value, resolved, context);
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

// io.write and io.print, whose print writes a string or a character itself,
// and their _line forms.
extern "C" void merk_write(Word typeInfo, Word value, Word flags) {
  Out out{stdout};
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
}

// string.string: what io.write writes, as a string.
extern "C" char *merk_string_of(Word typeInfo, Word value) {
  Out out;
  WriteTyped(out, value, Resolve(typeInfo, nullptr), kAnyPriority);
  return out.Take();
}
