//===-- unittests/Parser/ProvenanceTest.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/provenance.h"
#include "gtest/gtest.h"
#include "merkurius/Parser/source.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>

using namespace merkurius::parser;

namespace {

const SourceFile &add(
    AllSources &all, llvm::StringRef text, std::string path = "test.m") {
  return *all.AddBuffer(path, llvm::MemoryBuffer::getMemBufferCopy(text, path));
}

void expectPosition(
    const SourceFile &file, std::size_t offset, int line, int column) {
  SourcePosition pos{file.GetSourcePosition(offset)};
  EXPECT_EQ(pos.line, line) << "at offset " << offset;
  EXPECT_EQ(pos.column, column) << "at offset " << offset;
}

TEST(SourceFileTest, Positions) {
  AllSources all;
  const SourceFile &file{add(all, "ab\ncd")};
  EXPECT_EQ(file.lines(), 2u);
  expectPosition(file, 0, 1, 1);
  expectPosition(file, 2, 1, 3); // the newline
  expectPosition(file, 3, 2, 1);
  expectPosition(file, 5, 2, 3); // the end of the file
}

TEST(SourceFileTest, FinalNewlineStartsNoLine) {
  AllSources all;
  const SourceFile &file{add(all, "ab\n")};
  EXPECT_EQ(file.lines(), 1u);
  expectPosition(file, 3, 1, 4);
}

TEST(SourceFileTest, Empty) {
  AllSources all;
  const SourceFile &file{add(all, "")};
  EXPECT_EQ(file.bytes(), 0u);
  EXPECT_EQ(file.lines(), 1u);
  expectPosition(file, 0, 1, 1);
  EXPECT_EQ(all.GetProvenanceRange(file).size(), 1u);
}

TEST(SourceFileTest, KeepsBytesExactly) {
  AllSources all;
  const SourceFile &file{add(all, "a.\r\nb.\r\n")};
  EXPECT_EQ(file.content(), "a.\r\nb.\r\n");
  EXPECT_EQ(file.lines(), 2u);
  EXPECT_EQ(file.GetLine(1), "a.");
  EXPECT_EQ(file.GetLine(2), "b.");
}

TEST(SourceFileTest, ByteOrderMark) {
  AllSources all;
  const SourceFile &file{add(all, "\xef\xbb\xbf:- module m.\n")};
  EXPECT_TRUE(file.hasByteOrderMark());
  EXPECT_EQ(file.content(), ":- module m.\n");
  expectPosition(file, 0, 1, 1);
}

TEST(SourceFileTest, LineDirective) {
  AllSources all;
  SourceFile &file{const_cast<SourceFile &>(add(all, "a\nb\nc\nd\n"))};
  const std::string &original{file.SavePath("original.y")};
  file.LineDirective(3, original, 100);
  SourcePosition pos{file.GetSourcePosition(file.GetLineStartOffset(4))};
  EXPECT_EQ(pos.line, 101);
  EXPECT_EQ(pos.trueLineNumber, 4);
  EXPECT_EQ(*pos.path, "original.y");
  EXPECT_EQ(file.GetSourcePosition(0).line, 1);
}

TEST(AllSourcesTest, CharBlocksAndProvenances) {
  AllSources all;
  const SourceFile &a{add(all, "foo.\n", "a.m")};
  const SourceFile &b{add(all, "bar.\n", "b.m")};
  ProvenanceRange ra{all.GetProvenanceRange(a)};
  ProvenanceRange rb{all.GetProvenanceRange(b)};
  EXPECT_EQ(ra.size(), a.bytes() + 1);
  EXPECT_TRUE(ra.ImmediatelyPrecedes(rb));

  CharBlock bar{b.content().data(), 3};
  std::optional<ProvenanceRange> range{all.GetProvenanceRange(bar)};
  ASSERT_TRUE(range);
  EXPECT_EQ(range->start(), rb.start());
  EXPECT_EQ(all.GetPath(range->start()), "b.m");
  EXPECT_EQ(all[range->start() + std::size_t{2}], 'r');
  std::optional<CharBlock> back{all.GetCharBlock(*range)};
  ASSERT_TRUE(back);
  EXPECT_EQ(back->data(), bar.data());
  EXPECT_EQ(*back, "bar");

  // An empty block at the end of a file has the file's end provenance.
  CharBlock end{a.content().end(), std::size_t{0}};
  std::optional<ProvenanceRange> endRange{all.GetProvenanceRange(end)};
  ASSERT_TRUE(endRange);
  EXPECT_EQ(endRange->start(), ra.Last());
  EXPECT_EQ(all.GetSourcePosition(ra.Last())->line, 1);
  EXPECT_EQ(all.GetSourcePosition(ra.Last())->column, 6);

  std::string elsewhere{"foo"};
  EXPECT_FALSE(all.GetProvenanceRange(CharBlock{elsewhere}));
}

TEST(AllSourcesTest, EmitMessageCaret) {
  AllSources all;
  // Each of å, ä and ö is two bytes in UTF-8 but one column on screen.
  const SourceFile &file{
      add(all, "p :-\n\tX = \"\xc3\xa5\xc3\xa4\xc3\xb6\" ++ y.\n")};
  std::size_t y{file.content().find("y.")};
  ProvenanceRange range{all.GetProvenanceRange(file).start() + y, 1};
  std::string out;
  llvm::raw_string_ostream os{out};
  all.EmitMessage(os, range, "undefined variable",
      "error: ", llvm::raw_ostream::RED, /*echoSourceLine=*/true);
  EXPECT_EQ(out,
      "test.m:2:18: error: undefined variable\n"
      "  \tX = \"\xc3\xa5\xc3\xa4\xc3\xb6\" ++ y.\n"
      "  \t             ^\n");
}

} // namespace
