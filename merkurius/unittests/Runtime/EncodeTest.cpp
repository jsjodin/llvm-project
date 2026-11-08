//===-- unittests/Runtime/EncodeTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SHA-256 (runtime/sha256.cpp) against FIPS 180-4's examples and the edges of
// its padding, HMAC-SHA-256 against RFC 4231's test cases, and the wire format (runtime/encode.cpp) on descriptors the
// test makes, as code generation would: exact bytes, round trips, random
// bytes that must be rejected or decode to what encodes to them again, and
// fingerprints computed independently from the description that encode.cpp
// gives. test/CodeGen/encode-checks.m tests the format on the descriptors of
// a Mercury program.
//
//===----------------------------------------------------------------------===//

#include "encode.h"
#include "sha256.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace merkurius::runtime;

namespace {

std::string Hex(const std::uint8_t *bytes, std::size_t size) {
  static const char kDigits[]{"0123456789abcdef"};
  std::string hex;
  for (std::size_t i{0}; i < size; ++i) {
    hex += kDigits[bytes[i] >> 4];
    hex += kDigits[bytes[i] & 15];
  }
  return hex;
}

std::string Digest(const std::string &message, std::size_t piece = 0) {
  Sha256 sha;
  if (piece == 0) {
    sha.Update(message.data(), message.size());
  } else {
    for (std::size_t at{0}; at < message.size(); at += piece) {
      sha.Update(message.data() + at,
          piece < message.size() - at ? piece : message.size() - at);
    }
  }
  std::uint8_t digest[Sha256::kDigestBytes];
  sha.Finish(digest);
  return Hex(digest, sizeof digest);
}

TEST(Sha256Test, FipsExamples) {
  EXPECT_EQ(Digest(""),
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(Digest("abc"),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(Digest("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  EXPECT_EQ(Digest("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
                   "hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"),
      "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
  EXPECT_EQ(Digest(std::string(1000000, 'a')),
      "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

// Lengths about where the padding takes a second block.
TEST(Sha256Test, PaddingEdges) {
  struct Case {
    std::size_t length;
    const char *digest;
  };
  const Case kCases[]{
      {55, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"},
      {56, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"},
      {63, "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34"},
      {64, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"},
      {65, "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0"},
      {119, "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb"},
      {120, "2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c"},
  };
  for (const Case &c : kCases) {
    EXPECT_EQ(Digest(std::string(c.length, 'a')), c.digest) << c.length;
  }
}

std::string Bytes(const char *hex) {
  std::string bytes;
  for (const char *at{hex}; at[0] && at[1]; at += 2) {
    bytes += static_cast<char>(std::stoi(std::string(at, 2), nullptr, 16));
  }
  return bytes;
}

std::string Hmac(const std::string &key, const std::string &data) {
  std::uint8_t mac[Sha256::kDigestBytes];
  HmacSha256(key.data(), key.size(), data.data(), data.size(), mac);
  return Hex(mac, sizeof mac);
}

// RFC 4231, section 4: keys shorter and longer than a block, and data
// longer than a block. Test case 5's MAC is truncated to 128 bits.
TEST(HmacSha256Test, Rfc4231) {
  EXPECT_EQ(Hmac(std::string(20, '\x0b'), "Hi There"),
      "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
  EXPECT_EQ(Hmac("Jefe", "what do ya want for nothing?"),
      "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
  EXPECT_EQ(Hmac(std::string(20, '\xaa'), std::string(50, '\xdd')),
      "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe");
  EXPECT_EQ(Hmac(Bytes("0102030405060708090a0b0c0d0e0f10111213141516171819"),
                std::string(50, '\xcd')),
      "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b");
  EXPECT_EQ(Hmac(std::string(20, '\x0c'), "Test With Truncation").substr(0, 32),
      "a3b6167473100ee06e0c796c2955552b");
  EXPECT_EQ(Hmac(std::string(131, '\xaa'),
                "Test Using Larger Than Block-Size Key - Hash Key First"),
      "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
  EXPECT_EQ(Hmac(std::string(131, '\xaa'),
                "This is a test using a larger than block-size key and a "
                "larger than block-size data. The key needs to be hashed "
                "before being used by the HMAC algorithm."),
      "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2");
}

// The same digest whatever the pieces Update is given.
TEST(Sha256Test, Pieces) {
  std::string message(1000, 'x');
  for (std::size_t i{0}; i < message.size(); ++i) {
    message[i] = static_cast<char>(i * 7);
  }
  std::string whole{Digest(message)};
  for (std::size_t piece : {1, 3, 55, 63, 64, 65, 999}) {
    EXPECT_EQ(Digest(message, piece), whole) << piece;
  }
}

//===----------------------------------------------------------------------===//
// Descriptors, as LowerTypeCtors makes them.

const TypeCtor kInt{Int, 0, 0, 0, "", "int", nullptr, nullptr, nullptr, {},
    nullptr, nullptr};
const TypeCtor kString{String, 0, 0, 0, "", "string", nullptr, nullptr,
    nullptr, {}, nullptr, nullptr};
const TypeCtor kArray{Array, 0, 1, 0, "array", "array", nullptr, nullptr,
    nullptr, {}, nullptr, nullptr};

constexpr Word kParameter0{(Word{0} << kTagBits) | kVariableTag};

extern const TypeCtor kList;
// list(T), a field's type, as a cell of the descriptor and T.
const Word kListOfParameter[2]{
    reinterpret_cast<Word>(&kList), kParameter0};
const Word kConsFields[2]{
    kParameter0, reinterpret_cast<Word>(kListOfParameter) + 1};
const Functor kListFunctors[2]{
    {"[]", 0, 0, 0, -1, 0, nullptr},
    {"[|]", 2, 1, 0, -1, 0, kConsFields},
};
const std::uint16_t kListConstants[1]{0};
const TypeCtor kList{Cells, 0, 1, 2, "list", "list", nullptr, nullptr,
    kListFunctors,
    {kConstantTag, 1, kConstantTag, kConstantTag, kConstantTag, kConstantTag,
        kConstantTag, kConstantTag},
    kListConstants, nullptr};

const Word kListOfIntCell[2]{
    reinterpret_cast<Word>(&kList), reinterpret_cast<Word>(&kInt)};
const Word kArrayOfStringCell[2]{
    reinterpret_cast<Word>(&kArray), reinterpret_cast<Word>(&kString)};

const TypeCtor kChar{Char, 0, 0, 0, "", "character", nullptr, nullptr, nullptr,
    {}, nullptr, nullptr};
const Word kArrayOfIntCell[2]{
    reinterpret_cast<Word>(&kArray), reinterpret_cast<Word>(&kInt)};
const Word kArrayOfCharCell[2]{
    reinterpret_cast<Word>(&kArray), reinterpret_cast<Word>(&kChar)};

Word ArrayOfInt() { return reinterpret_cast<Word>(kArrayOfIntCell) + 1; }
Word ArrayOfChar() { return reinterpret_cast<Word>(kArrayOfCharCell) + 1; }

Word IntType() { return reinterpret_cast<Word>(&kInt); }
Word StringType() { return reinterpret_cast<Word>(&kString); }
Word ListOfInt() { return reinterpret_cast<Word>(kListOfIntCell) + 1; }
Word ArrayOfString() { return reinterpret_cast<Word>(kArrayOfStringCell) + 1; }

// A list of ints, as code generation makes one: [] is the word 0, and a
// cons a cell of two words, tagged 1.
Word MakeList(const std::vector<Word> &elements, std::vector<Word *> &cells) {
  Word list{0};
  for (auto it{elements.rbegin()}; it != elements.rend(); ++it) {
    Word *cell{new Word[2]{*it, list}};
    cells.push_back(cell);
    list = reinterpret_cast<Word>(cell) + 1;
  }
  return list;
}

std::vector<Word> ListElements(Word list) {
  std::vector<Word> elements;
  while (list != 0) {
    EXPECT_EQ(list & kTagMask, 1u);
    const Word *cell{reinterpret_cast<const Word *>(list - 1)};
    elements.push_back(cell[0]);
    list = cell[1];
  }
  return elements;
}

std::string EncodeHex(Word value, Word type) {
  Encoded encoded;
  EXPECT_EQ(Encode(value, type, encoded), nullptr);
  std::string hex{Hex(encoded.bytes, encoded.size)};
  std::free(encoded.bytes);
  return hex;
}

TEST(EncodeTest, ExactBytes) {
  EXPECT_EQ(EncodeHex(Word{0x0102030405060708}, IntType()),
      "0807060504030201");
  EXPECT_EQ(EncodeHex(reinterpret_cast<Word>("héllo, world"), StringType()),
      "0d00000000000000"
      "68c3a96c6c6f2c20776f726c64000000");
  std::vector<Word *> cells;
  EXPECT_EQ(EncodeHex(MakeList({1, 2}, cells), ListOfInt()),
      "0100000000000000"
      "0100000000000000"
      "0100000000000000"
      "0200000000000000"
      "0000000000000000");
  for (Word *cell : cells) {
    delete[] cell;
  }
}

// An array of ints is its elements as they are in memory.
TEST(EncodeTest, ArrayOfIntExactBytes) {
  const Word array[3]{2, 0x0102030405060708, 9};
  EXPECT_EQ(EncodeHex(reinterpret_cast<Word>(array), ArrayOfInt()),
      "0200000000000000"
      "0807060504030201"
      "0900000000000000");
  const Word empty[1]{0};
  EXPECT_EQ(EncodeHex(reinterpret_cast<Word>(empty), ArrayOfInt()),
      "0000000000000000");
}

TEST(EncodeTest, ArrayOfIntRoundTrip) {
  std::vector<Word> array{1000};
  for (Word i{1}; i <= 1000; ++i) {
    array.push_back(i * 0x9e3779b97f4a7c15);
  }
  Encoded encoded;
  ASSERT_EQ(Encode(reinterpret_cast<Word>(array.data()), ArrayOfInt(), encoded),
      nullptr);
  Fragment fragment;
  Word value;
  ASSERT_EQ(Decode(encoded.bytes, encoded.size, ArrayOfInt(), fragment, value),
      nullptr);
  std::free(encoded.bytes);
  const Word *decoded{reinterpret_cast<const Word *>(value)};
  for (Word i{0}; i <= 1000; ++i) {
    EXPECT_EQ(decoded[i], array[i]);
  }
}

// An array of characters is read an element at a time, to check each.
TEST(EncodeTest, ArrayOfCharIsChecked) {
  std::uint8_t bytes[16]{};
  bytes[0] = 1;
  bytes[8 + 2] = 0x11;
  Fragment fragment;
  Word value;
  EXPECT_STREQ(Decode(bytes, sizeof bytes, ArrayOfChar(), fragment, value),
      "a character is not a code point");
}

TEST(EncodeTest, ListRoundTrip) {
  std::vector<Word> elements;
  for (Word i{0}; i < 1000; ++i) {
    elements.push_back(i * 0x9e3779b97f4a7c15);
  }
  std::vector<Word *> cells;
  Word list{MakeList(elements, cells)};
  Encoded encoded;
  ASSERT_EQ(Encode(list, ListOfInt(), encoded), nullptr);
  EXPECT_EQ(encoded.size, 8u * (1 + 2 * elements.size()));
  Fragment fragment;
  Word decoded;
  ASSERT_EQ(Decode(encoded.bytes, encoded.size, ListOfInt(), fragment,
                decoded),
      nullptr);
  EXPECT_EQ(ListElements(decoded), elements);
  std::free(encoded.bytes);
  for (Word *cell : cells) {
    delete[] cell;
  }
}

// Random items: small numbers, mostly, so that lengths and functor numbers
// are often in range, letters, as a string's bytes, and any 64 bits; and
// now and then a few stray bytes.
std::vector<std::uint8_t> RandomItems(std::mt19937 &random) {
  std::vector<std::uint8_t> bytes;
  std::size_t items{random() % 12};
  for (std::size_t i{0}; i < items; ++i) {
    unsigned kind{static_cast<unsigned>(random() % 8)};
    std::uint64_t item{kind < 5 ? random() % 4
            : kind < 7         ? 0
                               : (std::uint64_t{random()} << 32) | random()};
    for (int b{0}; b < 8; ++b) {
      bytes.push_back(kind == 5 || kind == 6
              ? static_cast<std::uint8_t>(
                    b < static_cast<int>(random() % 9) ? 'a' + random() % 26 : 0)
              : static_cast<std::uint8_t>(item >> (8 * b)));
    }
  }
  if (random() % 10 == 0) {
    for (std::size_t i{random() % 8}; i != 0; --i) {
      bytes.push_back(static_cast<std::uint8_t>(random()));
    }
  }
  return bytes;
}

// Whatever the bytes, the decoder rejects them or gives a value that
// encodes to them again: the encoding has one form for each value.
void RandomBytes(Word type, int count, unsigned seed) {
  std::mt19937 random{seed};
  int decoded{0};
  for (int n{0}; n < count; ++n) {
    std::vector<std::uint8_t> bytes{RandomItems(random)};
    std::size_t size{bytes.size()};
    Fragment fragment;
    Word value;
    if (Decode(bytes.data(), bytes.size(), type, fragment, value) != nullptr) {
      continue;
    }
    ++decoded;
    Encoded encoded;
    ASSERT_EQ(Encode(value, type, encoded), nullptr);
    EXPECT_EQ(Hex(encoded.bytes, encoded.size), Hex(bytes.data(), size));
    std::free(encoded.bytes);
  }
  // Enough decode for the check to mean something.
  EXPECT_GT(decoded, count / 100);
}

TEST(EncodeTest, RandomBytesAsListOfInt) { RandomBytes(ListOfInt(), 100000, 1); }

TEST(EncodeTest, RandomBytesAsArrayOfString) {
  RandomBytes(ArrayOfString(), 100000, 2);
}

TEST(EncodeTest, HugeLengths) {
  std::uint8_t bytes[16]{};
  bytes[7] = 0x80;
  Fragment fragment;
  Word value;
  EXPECT_STREQ(Decode(bytes, sizeof bytes, ArrayOfString(), fragment, value),
      "an array is longer than the bytes left");
  EXPECT_STREQ(Decode(bytes, sizeof bytes, StringType(), fragment, value),
      "a string is longer than the bytes left");
  // The longest length that does not overflow when rounded up to 8.
  for (int i{0}; i < 8; ++i) {
    bytes[i] = 0xff;
  }
  EXPECT_STREQ(Decode(bytes, sizeof bytes, StringType(), fragment, value),
      "a string is longer than the bytes left");
}

//===----------------------------------------------------------------------===//
// Fingerprints, against SHA-256 of the description that encode.cpp gives,
// written out here.

std::string Item(Word item) {
  std::string bytes;
  for (int i{0}; i < 8; ++i) {
    bytes += static_cast<char>(item >> (8 * i));
  }
  return bytes;
}

std::string Text(const std::string &text) { return Item(text.size()) + text; }

std::string FingerprintHex(Word type) {
  std::uint8_t fingerprint[kFingerprintBytes];
  Fingerprint(type, fingerprint);
  return Hex(fingerprint, sizeof fingerprint);
}

TEST(FingerprintTest, Int) {
  std::string description{Item('T') + Text("") + Text("int") + Item(0)};
  EXPECT_EQ(FingerprintHex(IntType()), Digest(description).substr(0, 32));
}

TEST(FingerprintTest, ListOfInt) {
  std::string listOfParameter{
      Item('T') + Text("list") + Text("list") + Item(1) + Item('P') + Item(0)};
  std::string description{Item('T') + Text("list") + Text("list") + Item(1) +
      Item('T') + Text("") + Text("int") + Item(0) +
      // list's definition, the first time list is met.
      Item('D') + Item(2) + Text("[]") + Item(0) + Item(0) + Text("[|]") +
      Item(2) + Item(0) + Item('P') + Item(0) + listOfParameter};
  EXPECT_EQ(FingerprintHex(ListOfInt()), Digest(description).substr(0, 32));
  EXPECT_NE(FingerprintHex(ListOfInt()), FingerprintHex(IntType()));
  EXPECT_NE(FingerprintHex(ListOfInt()), FingerprintHex(ArrayOfString()));
}

} // namespace
