//===-- runtime/string.cpp - Strings --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What library/string.m does on bytes. A string is UTF-8 ending in a NUL, as
// in mmc's C grades (library manual, string), and its indices count bytes. A
// code unit that is not part of a well-formed sequence reads as U+FFFD, one
// for each such byte. New strings are allocated on the heap, as cells are.
//
//===----------------------------------------------------------------------===//

#include "heap.h"
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using merkurius::runtime::Allocate;

constexpr std::int32_t kReplacement{0xFFFD};

std::int64_t Clamp(std::int64_t value, std::int64_t low, std::int64_t high) {
  return value < low ? low : value > high ? high : value;
}

bool IsContinuation(unsigned char byte) { return (byte & 0xC0) == 0x80; }

// The code point at `offset`, and where the next starts.
std::int32_t Decode(const char *text, std::int64_t offset, std::int64_t *next) {
  const auto *s{reinterpret_cast<const unsigned char *>(text) + offset};
  unsigned char lead{s[0]};
  *next = offset + 1;
  if (lead < 0x80) {
    return lead;
  }
  int length;
  std::int32_t code;
  std::int32_t least;
  if ((lead & 0xE0) == 0xC0) {
    length = 2;
    code = lead & 0x1F;
    least = 0x80;
  } else if ((lead & 0xF0) == 0xE0) {
    length = 3;
    code = lead & 0x0F;
    least = 0x800;
  } else if ((lead & 0xF8) == 0xF0) {
    length = 4;
    code = lead & 0x07;
    least = 0x10000;
  } else {
    return kReplacement;
  }
  for (int i{1}; i < length; ++i) {
    // The NUL that ends the string is not a continuation byte, so this
    // stops at it.
    if (!IsContinuation(s[i])) {
      return kReplacement;
    }
    code = (code << 6) | (s[i] & 0x3F);
  }
  if (code < least || code > 0x10FFFF || (code >= 0xD800 && code <= 0xDFFF)) {
    return kReplacement;
  }
  *next = offset + length;
  return code;
}

char *NewString(std::size_t bytes) {
  // Rounded up to a word, so that the cells after it stay aligned.
  auto *text{static_cast<char *>(Allocate((bytes + 8) & ~std::size_t{7}))};
  text[bytes] = '\0';
  return text;
}

char *Copy(const char *from, std::size_t bytes) {
  char *text{NewString(bytes)};
  std::memcpy(text, from, bytes);
  return text;
}

bool IsWhitespace(unsigned char byte) {
  return byte == ' ' || (byte >= '\t' && byte <= '\r');
}

// What snprintf writes, as a new string.
__attribute__((format(printf, 1, 2))) char *Print(const char *format, ...) {
  std::va_list arguments;
  va_start(arguments, format);
  int bytes{std::vsnprintf(nullptr, 0, format, arguments)};
  va_end(arguments);
  if (bytes < 0) {
    // A width or precision too large for C's int: no string could hold it.
    std::abort();
  }
  char *text{NewString(static_cast<std::size_t>(bytes))};
  va_start(arguments, format);
  std::vsnprintf(text, static_cast<std::size_t>(bytes) + 1, format, arguments);
  va_end(arguments);
  return text;
}

// A string.format conversion's C format: its flags, of `-+ 0#`, its width
// and precision, each -1 when absent, a length modifier and the conversion.
void ConversionFormat(char *format, const char *flags, std::int64_t width,
    std::int64_t precision, const char *length, char conversion) {
  if (width > INT_MAX || precision > INT_MAX) {
    std::abort();
  }
  char *end{format};
  *end++ = '%';
  // Each flag once, however often it was written, which C reads the same.
  for (const char *flag{"-+ 0#"}; *flag; ++flag) {
    if (std::strchr(flags, *flag)) {
      *end++ = *flag;
    }
  }
  if (width >= 0) {
    end += std::sprintf(end, "%d", static_cast<int>(width));
  }
  if (precision >= 0) {
    end += std::sprintf(end, ".%d", static_cast<int>(precision));
  }
  for (; *length; ++length) {
    *end++ = *length;
  }
  *end++ = conversion;
  *end = '\0';
}

// Text padded with spaces to `width` code points, on the left unless the
// flags have `-`.
char *Pad(const char *text, std::size_t bytes, std::size_t count,
    const char *flags, std::int64_t width) {
  std::size_t padding{
      width > static_cast<std::int64_t>(count) ? width - count : 0};
  char *padded{NewString(bytes + padding)};
  bool left{std::strchr(flags, '-') != nullptr};
  char *start{left ? padded : padded + padding};
  std::memcpy(start, text, bytes);
  std::memset(left ? padded + bytes : padded, ' ', padding);
  return padded;
}

} // namespace

// The number of bytes of a code point's UTF-8, or 0 for one that no string
// may hold: NUL, a surrogate, or one beyond U+10FFFF.
extern "C" std::int64_t merk_char_bytes(std::int32_t code) {
  if (code <= 0 || (code >= 0xD800 && code <= 0xDFFF) || code > 0x10FFFF) {
    return 0;
  }
  return code < 0x80 ? 1 : code < 0x800 ? 2 : code < 0x10000 ? 3 : 4;
}

// A new string of `bytes` bytes, to be filled in, and its NUL.
extern "C" char *merk_string_new(std::int64_t bytes) {
  return NewString(static_cast<std::size_t>(bytes));
}

// Writes a code point's UTF-8 at `offset`, which merk_char_bytes has
// accepted, and gives the offset after it.
extern "C" std::int64_t merk_string_put(
    char *text, std::int64_t offset, std::int32_t code) {
  auto *s{reinterpret_cast<unsigned char *>(text) + offset};
  if (code < 0x80) {
    s[0] = static_cast<unsigned char>(code);
    return offset + 1;
  }
  if (code < 0x800) {
    s[0] = static_cast<unsigned char>(0xC0 | (code >> 6));
    s[1] = static_cast<unsigned char>(0x80 | (code & 0x3F));
    return offset + 2;
  }
  if (code < 0x10000) {
    s[0] = static_cast<unsigned char>(0xE0 | (code >> 12));
    s[1] = static_cast<unsigned char>(0x80 | ((code >> 6) & 0x3F));
    s[2] = static_cast<unsigned char>(0x80 | (code & 0x3F));
    return offset + 3;
  }
  s[0] = static_cast<unsigned char>(0xF0 | (code >> 18));
  s[1] = static_cast<unsigned char>(0x80 | ((code >> 12) & 0x3F));
  s[2] = static_cast<unsigned char>(0x80 | ((code >> 6) & 0x3F));
  s[3] = static_cast<unsigned char>(0x80 | (code & 0x3F));
  return offset + 4;
}

// Copies a string's bytes to `offset` and gives the offset after them.
extern "C" std::int64_t merk_string_put_string(
    char *text, std::int64_t offset, const char *from) {
  std::size_t bytes{std::strlen(from)};
  std::memcpy(text + offset, from, bytes);
  return offset + static_cast<std::int64_t>(bytes);
}

// The code point at `offset`, a byte of the string, and where the next one
// starts.
extern "C" std::int32_t merk_string_decode(
    const char *text, std::int64_t offset, std::int64_t *next) {
  return Decode(text, offset, next);
}

// Where the code point that ends just before `offset` starts: at most three
// continuation bytes back, if they and their lead decode as one code point,
// and one byte back otherwise.
extern "C" std::int64_t merk_string_previous(
    const char *text, std::int64_t offset) {
  for (std::int64_t back{2}; back <= 4 && offset - back >= 0; ++back) {
    if (!IsContinuation(static_cast<unsigned char>(text[offset - back + 1]))) {
      break;
    }
    std::int64_t next;
    if (Decode(text, offset - back, &next) != kReplacement &&
        next == offset) {
      return offset - back;
    }
  }
  return offset - 1;
}

extern "C" std::int64_t merk_string_count_codepoints(const char *text) {
  std::int64_t count{0};
  for (std::int64_t offset{0}; text[offset] != '\0'; ++count) {
    Decode(text, offset, &offset);
  }
  return count;
}

extern "C" char *merk_string_append(const char *a, const char *b) {
  std::size_t aBytes{std::strlen(a)}, bBytes{std::strlen(b)};
  char *text{NewString(aBytes + bBytes)};
  std::memcpy(text, a, aBytes);
  std::memcpy(text + aBytes, b, bBytes);
  return text;
}

// The bytes from `start` to just before `end`, each clamped to the string.
extern "C" char *merk_string_between(
    const char *text, std::int64_t start, std::int64_t end) {
  std::int64_t length{static_cast<std::int64_t>(std::strlen(text))};
  start = Clamp(start, 0, length);
  end = Clamp(end, start, length);
  return Copy(text + start, static_cast<std::size_t>(end - start));
}

// The byte offset of the first occurrence of `pattern`, or -1.
extern "C" std::int64_t merk_string_search(
    const char *text, const char *pattern) {
  const char *found{std::strstr(text, pattern)};
  return found ? found - text : -1;
}

// Whether `pattern` occurs at `offset`.
extern "C" std::int64_t merk_string_at(
    const char *text, std::int64_t offset, const char *pattern) {
  std::size_t bytes{std::strlen(pattern)};
  return std::strlen(text) >= static_cast<std::size_t>(offset) + bytes &&
      std::memcmp(text + offset, pattern, bytes) == 0;
}

// ASCII letters in the other case; other code points are kept.
extern "C" char *merk_string_case(const char *text, std::int64_t upper) {
  std::size_t bytes{std::strlen(text)};
  char *result{Copy(text, bytes)};
  for (std::size_t i{0}; i < bytes; ++i) {
    char c{result[i]};
    if (upper && c >= 'a' && c <= 'z') {
      result[i] = static_cast<char>(c - 'a' + 'A');
    } else if (!upper && c >= 'A' && c <= 'Z') {
      result[i] = static_cast<char>(c - 'A' + 'a');
    }
  }
  return result;
}

// Without its leading and trailing ASCII whitespace, as chosen.
extern "C" char *merk_string_strip(
    const char *text, std::int64_t leading, std::int64_t trailing) {
  std::size_t start{0}, end{std::strlen(text)};
  if (leading) {
    while (start < end && IsWhitespace(static_cast<unsigned char>(text[start]))) {
      ++start;
    }
  }
  if (trailing) {
    while (end > start &&
        IsWhitespace(static_cast<unsigned char>(text[end - 1]))) {
      --end;
    }
  }
  return Copy(text + start, end - start);
}

extern "C" std::int64_t merk_char_is_whitespace(std::int32_t code) {
  return code < 0x80 && IsWhitespace(static_cast<unsigned char>(code));
}

// Every occurrence of `pattern` replaced, from the left, without overlaps; an
// empty pattern occurs before each code point and at the end.
extern "C" char *merk_string_replace_all(
    const char *text, const char *pattern, const char *replacement) {
  std::size_t bytes{std::strlen(text)}, patternBytes{std::strlen(pattern)},
      replacementBytes{std::strlen(replacement)};
  std::size_t count{0};
  if (patternBytes == 0) {
    count = static_cast<std::size_t>(merk_string_count_codepoints(text)) + 1;
  } else {
    for (const char *at{std::strstr(text, pattern)}; at;
        at = std::strstr(at + patternBytes, pattern)) {
      ++count;
    }
  }
  char *result{
      NewString(bytes + count * replacementBytes - count * patternBytes)};
  char *out{result};
  if (patternBytes == 0) {
    std::int64_t offset{0};
    for (;;) {
      std::memcpy(out, replacement, replacementBytes);
      out += replacementBytes;
      if (text[offset] == '\0') {
        break;
      }
      std::int64_t next;
      Decode(text, offset, &next);
      std::memcpy(out, text + offset, static_cast<std::size_t>(next - offset));
      out += next - offset;
      offset = next;
    }
    return result;
  }
  const char *from{text};
  for (const char *at{std::strstr(text, pattern)}; at;
      at = std::strstr(from, pattern)) {
    std::memcpy(out, from, static_cast<std::size_t>(at - from));
    out += at - from;
    std::memcpy(out, replacement, replacementBytes);
    out += replacementBytes;
    from = at + patternBytes;
  }
  std::memcpy(out, from, std::strlen(from));
  return result;
}

// A decimal int with an optional sign, and nothing else, that fits.
extern "C" std::int64_t merk_string_to_int(
    const char *text, std::int64_t *value) {
  const char *s{text};
  bool negative{false};
  if (*s == '-' || *s == '+') {
    negative = *s == '-';
    ++s;
  }
  if (*s == '\0') {
    return 0;
  }
  // Accumulated negatively, so that the least int fits.
  std::int64_t result{0};
  for (; *s != '\0'; ++s) {
    if (*s < '0' || *s > '9') {
      return 0;
    }
    std::int64_t digit{*s - '0'};
    if (__builtin_mul_overflow(result, 10, &result) ||
        __builtin_sub_overflow(result, digit, &result)) {
      return 0;
    }
  }
  if (!negative && __builtin_mul_overflow(result, -1, &result)) {
    return 0;
  }
  *value = result;
  return 1;
}

// A float as C's strtod reads it, the whole string, with no leading
// whitespace, which strtod would skip.
extern "C" std::int64_t merk_string_to_float(const char *text, double *value) {
  if (*text == '\0' || IsWhitespace(static_cast<unsigned char>(*text))) {
    return 0;
  }
  char *end{nullptr};
  double result{std::strtod(text, &end)};
  if (*end != '\0') {
    return 0;
  }
  *value = result;
  return 1;
}

extern "C" char *merk_string_from_int(std::int64_t value) {
  char buffer[32];
  int bytes{std::snprintf(buffer, sizeof buffer, "%lld",
      static_cast<long long>(value))};
  return Copy(buffer, static_cast<std::size_t>(bytes));
}

extern "C" void merk_format_float(double number, char *buffer);

extern "C" char *merk_string_from_float(double value) {
  char buffer[64];
  merk_format_float(value, buffer);
  return Copy(buffer, std::strlen(buffer));
}

// string.format's conversions of integers, d, i, o, x, X, u and p, of a value
// of `bits` bits: sign-extended for d and i, which only the signed types
// take, and as an unsigned value of its bits for the others, as C's are.
extern "C" char *merk_format_integer(const char *flags, std::int64_t width,
    std::int64_t precision, std::int32_t conversion, std::int64_t value,
    std::int64_t bits) {
  std::uint64_t mask{bits >= 64 ? ~std::uint64_t{0}
                                : (std::uint64_t{1} << bits) - 1};
  std::uint64_t unsignedValue{static_cast<std::uint64_t>(value) & mask};
  char format[64];
  char c{static_cast<char>(conversion)};
  if (c == 'p') {
    ConversionFormat(format, flags, width, precision, "", c);
    return Print(format, reinterpret_cast<void *>(unsignedValue));
  }
  ConversionFormat(format, flags, width, precision, "ll", c);
  if (c == 'd' || c == 'i') {
    return Print(format, static_cast<long long>(value));
  }
  return Print(format, static_cast<unsigned long long>(unsignedValue));
}

// string.format's conversions of floats, f, F, e, E, g and G. An infinity or
// a NaN is written as io.write_float writes it, in capitals for F, E and G,
// and only padded with spaces, as mmc does.
extern "C" char *merk_format_float_conversion(const char *flags,
    std::int64_t width, std::int64_t precision, std::int32_t conversion,
    double value) {
  char c{static_cast<char>(conversion)};
  if (std::isnan(value) || std::isinf(value)) {
    bool upper{c == 'F' || c == 'E' || c == 'G'};
    const char *text{std::isnan(value) ? (upper ? "NAN" : "nan")
            : value < 0                ? (upper ? "-INFINITY" : "-infinity")
                                       : (upper ? "INFINITY" : "infinity")};
    std::size_t bytes{std::strlen(text)};
    return Pad(text, bytes, bytes, flags, width);
  }
  char format[64];
  ConversionFormat(format, flags, width, precision, "", c);
  return Print(format, value);
}

// string.format's s and c: the text's first `precision` code points, or all
// of them for -1, padded to `width` code points with spaces, as mmc does.
extern "C" char *merk_format_text(const char *flags, std::int64_t width,
    std::int64_t precision, const char *text) {
  std::int64_t bytes{0};
  std::size_t count{0};
  while (text[bytes] && (precision < 0 ||
                            static_cast<std::int64_t>(count) < precision)) {
    Decode(text, bytes, &bytes);
    ++count;
  }
  return Pad(text, static_cast<std::size_t>(bytes), count, flags, width);
}
