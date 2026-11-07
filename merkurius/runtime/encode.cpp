//===-- runtime/encode.cpp - The wire format ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The encoding of a message for another node (docs/INPROGRESS.md, Remote
// handles and TCP, "The wire format"). A value is walked by its type_info, as
// the local copy walks it (mailbox.cpp), into items of 8 bytes,
// little-endian:
//
// - an integer of any size, or a character, as its value, a signed one
//   sign-extended; a float as its bits;
// - a string as its length in bytes, then its bytes, with zeros to a multiple
//   of 8;
// - an array as its length, then its elements;
// - a value of a type of constructors, a tuple's too, as its functor's number
//   in the order declared, then its arguments.
//
// So no value takes less than an item, and an array of numbers is its
// elements as they are in memory, on the little-endian machines that are
// targeted. Nothing depends on the word size or the tags. The decoder walks
// the receiver's type_info and checks every number, length, range and
// padding against it.
//
// A fingerprint is SHA-256, cut to 128 bits, of a description of the type:
// each type constructor by its module, which is empty for the builtin
// types, name and arguments, and the first
// time it is met, the definition of one of constructors: each functor's
// name, arity and arguments' types, with its type parameters by number.
//
// Not yet: existentially typed constructors and closures, which the local
// copy rejects too.
//
//===----------------------------------------------------------------------===//

#include "encode.h"
#include "sha256.h"
#include "walk.h"
#include <cstring>

namespace merkurius::runtime {

namespace {

constexpr Word kElement{(Word{0} << kTagBits) | kVariableTag};

std::int64_t SignExtend(Word value, unsigned bits) {
  return static_cast<std::int64_t>(value << (64 - bits)) >> (64 - bits);
}

bool IsTerm(const TypeCtor *ctor) {
  return ctor->kind == Enum || ctor->kind == Dummy || ctor->kind == NoTag ||
      ctor->kind == Cells;
}

std::size_t FunctorNumber(const Functor *functor, const TypeCtor *ctor) {
  return static_cast<std::size_t>(functor - ctor->functors);
}

//===----------------------------------------------------------------------===//
// Encoding.

class Writer {
public:
  explicit Writer(Encoded &out) : out_{out} {}

  void Item(Word item) {
    std::uint8_t *at{Reserve(8)};
    for (int i{0}; i < 8; ++i) {
      at[i] = static_cast<std::uint8_t>(item >> (8 * i));
    }
  }

  void Text(const char *text, std::size_t bytes) {
    Item(bytes);
    std::size_t padded{(bytes + 7) & ~std::size_t{7}};
    std::uint8_t *at{Reserve(padded)};
    std::memcpy(at, text, bytes);
    std::memset(at + bytes, 0, padded - bytes);
  }

  void Encode(Word value, Word type, const Environment *environment) {
    EncodeTyped(value, Resolve(type, environment));
  }

  void EncodeTyped(Word value, const Type &type);

  const char *error{nullptr};

private:
  std::uint8_t *Reserve(std::size_t bytes) {
    if (out_.size + bytes > capacity_) {
      capacity_ = capacity_ * 2 > out_.size + bytes ? capacity_ * 2
                                                    : out_.size + bytes + 64;
      auto *grown{static_cast<std::uint8_t *>(Malloc(capacity_))};
      if (out_.size != 0) {
        std::memcpy(grown, out_.bytes, out_.size);
      }
      std::free(out_.bytes);
      out_.bytes = grown;
    }
    std::uint8_t *at{out_.bytes + out_.size};
    out_.size += bytes;
    return at;
  }

  void EncodeTerm(Word value, const Type &type);

  Encoded &out_;
  std::size_t capacity_{0};
};

void Writer::EncodeTyped(Word value, const Type &type) {
  if (error) {
    return;
  }
  switch (type.ctor->kind) {
  case Int:
  case Int64:
  case Uint:
  case Uint64:
  case Float:
  case Char:
  case Enum:
    return Item(value);
  case Int8:
    return Item(static_cast<Word>(SignExtend(value, 8)));
  case Int16:
    return Item(static_cast<Word>(SignExtend(value, 16)));
  case Int32:
    return Item(static_cast<Word>(SignExtend(value, 32)));
  case Uint8:
    return Item(value & 0xff);
  case Uint16:
    return Item(value & 0xffff);
  case Uint32:
    return Item(value & 0xffffffff);
  case Dummy:
    return Item(0);
  case String: {
    const char *text{reinterpret_cast<const char *>(value)};
    return Text(text, std::strlen(text));
  }
  case Array: {
    const Word *array{reinterpret_cast<const Word *>(value)};
    Item(array[0]);
    for (Word i{0}; i < array[0] && !error; ++i) {
      Encode(array[1 + i], kElement, &type.environment);
    }
    return;
  }
  case NoTag:
  case Tuple:
  case Cells:
    return EncodeTerm(value, type);
  case Closure:
    error = "a closure cannot be sent to another node yet";
    return;
  case Void:
    // Nothing has a value of this type.
    std::abort();
  }
  error = "a value of an unknown kind of type";
}

// A field of the term's own type is encoded in the loop rather than by a
// call, so that the length of a list uses no stack.
void Writer::EncodeTerm(Word value, const Type &type) {
  for (;;) {
    const Word *fields{nullptr};
    Word term{value};
    const Functor *functor{FunctorOf(term, type.ctor, fields)};
    Item(FunctorNumber(functor, type.ctor));
    if (functor->hidden != 0) {
      error = "a value with an existentially typed constructor cannot be "
              "sent to another node yet";
      return;
    }
    unsigned arity{functor->arity};
    if (!fields || arity == 0) {
      return;
    }
    if (fields == &term) {
      // A notag's or a tuple of one's only field, which is the value.
      return Encode(value, functor->fields[0], &type.environment);
    }
    for (unsigned i{0}; i + 1 < arity && !error; ++i) {
      Encode(fields[i], functor->fields[i], &type.environment);
    }
    Word last{functor->fields[arity - 1]};
    if (error) {
      return;
    }
    if (!IsSameType(last, type) || type.ctor->kind == Tuple) {
      return Encode(fields[arity - 1], last, &type.environment);
    }
    value = fields[arity - 1];
  }
}

//===----------------------------------------------------------------------===//
// Decoding.

class Reader {
public:
  Reader(const std::uint8_t *bytes, std::size_t size, Fragment &fragment)
      : next_{bytes}, left_{size}, fragment_{fragment} {}

  bool Decode(Word type, const Environment *environment, Word &value) {
    if (++depth_ > kMaxDecodeDepth) {
      return Fail("the value nests too deeply");
    }
    bool ok{DecodeTyped(Resolve(type, environment), value)};
    --depth_;
    return ok;
  }

  bool DecodeTyped(const Type &type, Word &value);

  std::size_t Left() const { return left_; }

  const char *error{nullptr};

private:
  bool Fail(const char *why) {
    if (!error) {
      error = why;
    }
    return false;
  }

  bool Item(Word &item) {
    if (left_ < 8) {
      return Fail("the bytes end within a value");
    }
    item = 0;
    for (int i{0}; i < 8; ++i) {
      item |= Word{next_[i]} << (8 * i);
    }
    next_ += 8;
    left_ -= 8;
    return true;
  }

  bool Signed(Word &value, unsigned bits) {
    if (!Item(value)) {
      return false;
    }
    if (SignExtend(value, bits) != static_cast<std::int64_t>(value)) {
      return Fail("an integer is out of its type's range");
    }
    return true;
  }

  bool Unsigned(Word &value, unsigned bits) {
    if (!Item(value)) {
      return false;
    }
    if ((value >> bits) != 0) {
      return Fail("an integer is out of its type's range");
    }
    return true;
  }

  bool Number(Word &number, const TypeCtor *ctor) {
    if (!Item(number)) {
      return false;
    }
    if (number >= ctor->functorCount) {
      return Fail("a functor's number is not one of its type's");
    }
    return true;
  }

  bool DecodeString(Word &value);
  bool DecodeTerm(const Type &type, Word &value);

  const std::uint8_t *next_;
  std::size_t left_;
  Fragment &fragment_;
  int depth_{0};
};

bool Reader::DecodeString(Word &value) {
  Word bytes;
  if (!Item(bytes)) {
    return false;
  }
  if (bytes > left_ || ((bytes + 7) & ~Word{7}) > left_) {
    return Fail("a string is longer than the bytes left");
  }
  std::size_t padded{static_cast<std::size_t>((bytes + 7) & ~Word{7})};
  if (std::memchr(next_, 0, bytes)) {
    return Fail("a string has a NUL");
  }
  for (std::size_t i{bytes}; i < padded; ++i) {
    if (next_[i] != 0) {
      return Fail("a string's padding is not zeros");
    }
  }
  char *text{static_cast<char *>(fragment_.Allocate(bytes + 1))};
  std::memcpy(text, next_, bytes);
  text[bytes] = '\0';
  next_ += padded;
  left_ -= padded;
  value = reinterpret_cast<Word>(text);
  return true;
}

bool Reader::DecodeTyped(const Type &type, Word &value) {
  switch (type.ctor->kind) {
  case Int:
  case Int64:
  case Uint:
  case Uint64:
  case Float:
    return Item(value);
  case Int8:
    return Signed(value, 8);
  case Int16:
    return Signed(value, 16);
  case Int32:
    return Signed(value, 32);
  case Uint8:
    return Unsigned(value, 8);
  case Uint16:
    return Unsigned(value, 16);
  case Uint32:
    return Unsigned(value, 32);
  case Char:
    if (!Item(value)) {
      return false;
    }
    if (value > 0x10ffff) {
      return Fail("a character is not a code point");
    }
    return true;
  case String:
    return DecodeString(value);
  case Array: {
    Word size;
    if (!Item(size)) {
      return false;
    }
    // Each element takes an item at least.
    if (size > left_ / 8) {
      return Fail("an array is longer than the bytes left");
    }
    auto *array{
        static_cast<Word *>(fragment_.Allocate(sizeof(Word) * (1 + size)))};
    array[0] = size;
    for (Word i{0}; i < size; ++i) {
      if (!Decode(kElement, &type.environment, array[1 + i])) {
        return false;
      }
    }
    value = reinterpret_cast<Word>(array);
    return true;
  }
  case Enum:
    return Number(value, type.ctor);
  case Dummy:
    if (!Number(value, type.ctor)) {
      return false;
    }
    value = 0;
    return true;
  case NoTag:
  case Tuple:
  case Cells:
    return DecodeTerm(type, value);
  case Closure:
    return Fail("a closure cannot be received from another node yet");
  case Void:
    return Fail("nothing has a value of type void");
  }
  return Fail("a value of an unknown kind of type");
}

// A term's cell made as code generation makes it: a constant's word, or a
// cell of its secondary tag, if it has one, and its arguments, tagged.
bool Reader::DecodeTerm(const Type &type, Word &value) {
  Word *slot{&value};
  for (;;) {
    Word number;
    if (!Number(number, type.ctor)) {
      return false;
    }
    const Functor *functor{&type.ctor->functors[number]};
    if (functor->hidden != 0) {
      return Fail("a value with an existentially typed constructor cannot "
                  "be received from another node yet");
    }
    unsigned arity{functor->arity};
    if (type.ctor->kind == NoTag ||
        (type.ctor->kind == Tuple && arity == 1)) {
      return Decode(functor->fields[0], &type.environment, *slot);
    }
    if (type.ctor->kind == Cells && arity == 0) {
      *slot = static_cast<Word>(functor->constant);
      return true;
    }
    unsigned before{type.ctor->kind == Cells && functor->secondary >= 0 ? 1u
                                                                       : 0u};
    // A tuple of no arguments is given a word, which nothing reads.
    unsigned words{before + arity != 0 ? before + arity : 1u};
    auto *cell{static_cast<Word *>(fragment_.Allocate(sizeof(Word) * words))};
    if (before != 0) {
      cell[0] = static_cast<Word>(functor->secondary);
    }
    *slot = reinterpret_cast<Word>(cell) + functor->tag;
    if (arity == 0) {
      return true;
    }
    Word *arguments{cell + before};
    for (unsigned i{0}; i + 1 < arity; ++i) {
      if (!Decode(functor->fields[i], &type.environment, arguments[i])) {
        return false;
      }
    }
    Word last{functor->fields[arity - 1]};
    if (!IsSameType(last, type) || type.ctor->kind == Tuple) {
      return Decode(last, &type.environment, arguments[arity - 1]);
    }
    slot = &arguments[arity - 1];
  }
}

//===----------------------------------------------------------------------===//
// Fingerprints.

class Describer {
public:
  void Item(Word item) {
    std::uint8_t bytes[8];
    for (int i{0}; i < 8; ++i) {
      bytes[i] = static_cast<std::uint8_t>(item >> (8 * i));
    }
    sha_.Update(bytes, 8);
  }

  void Text(const char *text) {
    std::size_t bytes{std::strlen(text)};
    Item(bytes);
    sha_.Update(text, bytes);
  }

  // A type, a type_info or a field's, whose type parameters stay numbers.
  void Describe(Word type) {
    switch (type & kTagMask) {
    case kVariableTag:
      Item('P');
      Item(type >> kTagBits);
      return;
    case kHiddenTag:
      Item('H');
      Item(type >> kTagBits);
      return;
    case 0:
      return Constructor(reinterpret_cast<const TypeCtor *>(type), nullptr, 0);
    default: {
      const Word *cell{reinterpret_cast<const Word *>(type - 1)};
      const auto *ctor{reinterpret_cast<const TypeCtor *>(cell[0])};
      bool counted{ctor->kind == Tuple || ctor->kind == Closure};
      return Constructor(ctor, cell + (counted ? 2 : 1),
          counted ? static_cast<std::size_t>(cell[1]) : ctor->arity);
    }
    }
  }

  void Finish(std::uint8_t fingerprint[kFingerprintBytes]) {
    std::uint8_t digest[Sha256::kDigestBytes];
    sha_.Finish(digest);
    std::memcpy(fingerprint, digest, kFingerprintBytes);
    std::free(defined_);
  }

private:
  void Constructor(
      const TypeCtor *ctor, const Word *arguments, std::size_t count) {
    Item('T');
    Text(ctor->module);
    Text(ctor->name);
    Item(count);
    for (std::size_t i{0}; i < count; ++i) {
      Describe(arguments[i]);
    }
    if (IsTerm(ctor) && !IsDefined(ctor)) {
      Define(ctor);
    }
  }

  // Types are told apart by name, not by descriptor, of which a program
  // may have more than one for a type.
  bool IsDefined(const TypeCtor *ctor) const {
    for (std::size_t i{0}; i < count_; ++i) {
      if (defined_[i]->arity == ctor->arity &&
          std::strcmp(defined_[i]->name, ctor->name) == 0 &&
          std::strcmp(defined_[i]->module, ctor->module) == 0) {
        return true;
      }
    }
    return false;
  }

  void Define(const TypeCtor *ctor) {
    if (count_ == capacity_) {
      capacity_ = capacity_ ? capacity_ * 2 : 16;
      auto *grown{static_cast<const TypeCtor **>(
          Malloc(sizeof(const TypeCtor *) * capacity_))};
      if (count_ != 0) {
        std::memcpy(grown, defined_, sizeof(const TypeCtor *) * count_);
      }
      std::free(defined_);
      defined_ = grown;
    }
    defined_[count_++] = ctor;
    Item('D');
    Item(ctor->functorCount);
    for (std::uint32_t i{0}; i < ctor->functorCount; ++i) {
      const Functor &functor{ctor->functors[i]};
      Text(functor.name);
      Item(functor.arity);
      Item(functor.hidden);
      for (std::uint32_t j{0}; j < functor.arity; ++j) {
        Describe(functor.fields[j]);
      }
    }
  }

  Sha256 sha_;
  const TypeCtor **defined_{nullptr};
  std::size_t count_{0};
  std::size_t capacity_{0};
};

} // namespace

const char *Encode(Word value, Word typeInfo, Encoded &encoded) {
  encoded = {};
  Writer writer{encoded};
  writer.EncodeTyped(value, Resolve(typeInfo, nullptr));
  if (writer.error) {
    std::free(encoded.bytes);
    encoded = {};
  }
  return writer.error;
}

const char *Decode(const std::uint8_t *bytes, std::size_t size, Word typeInfo,
    Fragment &fragment, Word &value) {
  Reader reader{bytes, size, fragment};
  if (!reader.Decode(typeInfo, nullptr, value)) {
    return reader.error;
  }
  if (reader.Left() != 0) {
    return "there are bytes after the value";
  }
  return nullptr;
}

void Fingerprint(Word typeInfo, std::uint8_t fingerprint[kFingerprintBytes]) {
  Describer describer;
  describer.Describe(typeInfo);
  describer.Finish(fingerprint);
}

} // namespace merkurius::runtime

//===----------------------------------------------------------------------===//
// For C: tests' foreign_procs, until connections use the encoding.

using namespace merkurius::runtime;

// The encoding of a value, from malloc, and its size; null, or why it cannot
// be encoded.
extern "C" const char *merk_encode(
    Word typeInfo, Word value, std::uint8_t **bytes, std::int64_t *size) {
  Encoded encoded;
  const char *error{Encode(value, typeInfo, encoded)};
  *bytes = encoded.bytes;
  *size = static_cast<std::int64_t>(encoded.size);
  return error;
}

// A value from an encoding, its cells in a fragment of their own; null, or
// why the bytes are not one.
extern "C" const char *merk_decode(
    Word typeInfo, const std::uint8_t *bytes, std::int64_t size, Word *value) {
  // Temporary (chunk 2, docs/INPROGRESS.md): as a local send's, the
  // fragment's chunks are never freed.
  Fragment fragment;
  return Decode(bytes, static_cast<std::size_t>(size), typeInfo, fragment,
      *value);
}

extern "C" void merk_fingerprint(Word typeInfo, std::uint8_t *fingerprint) {
  Fingerprint(typeInfo, fingerprint);
}
