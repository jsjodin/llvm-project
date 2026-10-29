//===-- tools/merkurius-term-fuzzer/merkurius-term-fuzzer.cpp ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A fuzz target for the lexer and the term reader. For any input it checks
// that neither crashes, with and without OpenMP directives, that the tokens
// unparse to the input byte for byte, and that each item's canonical form
// reads back as the same canonical form.
//
// Built normally, it is a program that runs the target on each file named
// on its command line, to replay a corpus. Built with -fsanitize=fuzzer
// (utils/fuzz/run-term-fuzzer.sh), libFuzzer provides main instead.
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/lexer.h"
#include "merkurius/Parser/message.h"
#include "merkurius/Parser/provenance.h"
#include "merkurius/Parser/source.h"
#include "merkurius/Parser/term-reader.h"
#include "merkurius/Parser/term.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

using namespace merkurius::parser;

namespace {

// Less than the reader's default, as sanitizers make each level's stack
// frames larger and the fuzzer runs on the main thread.
constexpr unsigned maxDepth{1000};

void Fail(const char *what, llvm::StringRef input, llvm::StringRef detail) {
  llvm::errs() << "merkurius-term-fuzzer: " << what << "\ninput: " << input
               << "\ndetail: " << detail << '\n';
  std::abort();
}

// The canonical form of each item that reads without messages, and an
// empty string for each other item.
std::vector<std::string> ReadCanonical(
    llvm::StringRef text, bool openMP, bool checkUnparse) {
  AllSources all;
  SourceFile *file{all.AddBuffer(
      "fuzz.m", llvm::MemoryBuffer::getMemBufferCopy(text, "fuzz.m"))};
  Messages messages;
  LexerOptions options;
  options.openMP = openMP;
  TokenSequence tokens{Lex(*file, options, messages)};
  if (checkUnparse) {
    std::string unparsed;
    llvm::raw_string_ostream os{unparsed};
    UnparseTokens(os, tokens);
    if (unparsed != text) {
      Fail("the tokens do not unparse to the input", text, unparsed);
    }
  }
  bool lexerErrors{messages.AnyFatalError()};
  TermArena arena;
  TermReader reader{tokens, arena, messages, maxDepth};
  std::vector<std::string> items;
  std::size_t count{messages.messages().size()};
  while (std::optional<Item> item{reader.ReadItem()}) {
    std::string canonical;
    if (item->term && !lexerErrors && messages.messages().size() == count) {
      llvm::raw_string_ostream os{canonical};
      PrintCanonical(os, *item->term);
    }
    count = messages.messages().size();
    items.push_back(std::move(canonical));
    arena.Reset();
  }
  return items;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(
    const std::uint8_t *data, std::size_t size) {
  llvm::StringRef input{reinterpret_cast<const char *>(data), size};
  for (bool openMP : {false, true}) {
    for (const std::string &canonical : ReadCanonical(input, openMP, true)) {
      if (canonical.empty()) {
        continue;
      }
      // Without directives, a canonical form is one item that prints the
      // same; '%$omp' is a quoted name there.
      std::string item{canonical + ".\n"};
      std::vector<std::string> again{ReadCanonical(item, false, false)};
      if (again.size() != 1 || again[0] != canonical) {
        Fail("a canonical form does not read back as itself", input,
            canonical + " became " + (again.empty() ? "nothing" : again[0]));
      }
    }
  }
  return 0;
}

#ifndef MERKURIUS_LIBFUZZER
int main(int argc, char **argv) {
  for (int j{1}; j < argc; ++j) {
    auto buffer{llvm::MemoryBuffer::getFile(argv[j])};
    if (!buffer) {
      llvm::errs() << "merkurius-term-fuzzer: cannot read " << argv[j] << '\n';
      return 1;
    }
    llvm::StringRef text{(*buffer)->getBuffer()};
    LLVMFuzzerTestOneInput(
        reinterpret_cast<const std::uint8_t *>(text.data()), text.size());
  }
  return 0;
}
#endif
