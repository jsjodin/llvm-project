//===-- tools/merkurius-term-fuzzer/merkurius-term-fuzzer.cpp ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A fuzz target for the lexer, the term reader, the converter, the
// unparser and desugaring. For any input, with and without OpenMP
// directives, it checks that none of them crashes; that the tokens, and the parse tree, unparse to
// the input byte for byte; that the exact unparser finds every token
// accounted for; that an input which converts without errors has a
// canonical layout that is its own canonical layout, and the same parse tree
// in that layout and with every operator term parenthesized; and that each
// item's canonical term reads back as the same canonical term.
//
// Built normally, it is a program that runs the target on each file named
// on its command line, to replay a corpus. Built with -fsanitize=fuzzer
// (utils/fuzz/run-term-fuzzer.sh), libFuzzer provides main instead.
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/convert.h"
#include "merkurius/Parser/lexer.h"
#include "merkurius/Parser/message.h"
#include "merkurius/Parser/provenance.h"
#include "merkurius/Parser/source.h"
#include "merkurius/Parser/term-reader.h"
#include "merkurius/Parser/term.h"
#include "merkurius/Parser/unparse.h"
#include "merkurius/Semantics/check-determinism.h"
#include "merkurius/Semantics/check-modes.h"
#include "merkurius/Semantics/check-purity.h"
#include "merkurius/Semantics/check-types.h"
#include "merkurius/Semantics/desugar.h"
#include "merkurius/Semantics/module-loader.h"
#include "merkurius/Semantics/resolve-names.h"
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
  while (std::optional<ItemTerm> item{reader.ReadItem()}) {
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

// A conversion of the input: its parse tree's dump and its unparsings.
struct Unparsed {
  bool ok; // converted without errors
  std::string tree, exact, canonical, parenthesized;
};

// Converts and unparses `text`, and fails the input if the exact unparser
// reports a problem.
Unparsed ConvertAndUnparse(llvm::StringRef text, bool openMP) {
  AllSources all;
  SourceFile *file{
      all.AddBuffer("fuzz.m", llvm::MemoryBuffer::getMemBufferCopy(text))};
  Messages messages;
  LexerOptions options;
  options.openMP = openMP;
  TokenSequence tokens{Lex(*file, options, messages)};
  ParseTreeArena arena;
  const ParseTree &tree{ConvertFile(tokens, arena, messages, maxDepth)};
  Unparsed result;
  result.ok = !messages.AnyFatalError();
  Messages problems;
  llvm::raw_string_ostream exact{result.exact};
  UnparseExact(exact, tree, tokens, UnparseLayout::Exact, result.ok, problems);
  if (!problems.messages().empty()) {
    Fail("the parse tree does not account for its tokens", text,
        problems.messages().front().ToString());
  }
  llvm::raw_string_ostream dump{result.tree};
  DumpParseTree(dump, tree);
  llvm::raw_string_ostream canonical{result.canonical};
  UnparseCanonical(canonical, tree, tokens);
  llvm::raw_string_ostream parenthesized{result.parenthesized};
  UnparseCanonical(parenthesized, tree, tokens, true);
  // Desugaring must not crash on whatever converted.
  merkurius::semantics::SemanticArena semanticArena;
  std::string form;
  llvm::raw_string_ostream formStream{form};
  const merkurius::semantics::SemanticForm &semanticForm{
      merkurius::semantics::Desugar(tree, tokens, semanticArena, messages)};
  merkurius::semantics::DumpSemanticForm(formStream, semanticForm);
  // So must name resolution. No imported module can be found, so only the
  // module's own names resolve.
  merkurius::semantics::ModuleLoader loader{all, "", messages};
  loader.LoadImports(tree, tokens, "/nonexistent");
  std::unique_ptr<merkurius::semantics::NameResolution> resolution{
      merkurius::semantics::ResolveNames(
          tree, tokens, semanticForm, loader, messages)};
  // And type checking, purity checking, mode checking and determinism
  // analysis.
  std::unique_ptr<merkurius::semantics::TypeChecking> checking{
      merkurius::semantics::CheckTypes(
          semanticForm, *resolution, tokens, messages)};
  merkurius::semantics::CheckPurity(
      tree, tokens, semanticForm, *resolution, *checking, messages);
  std::unique_ptr<merkurius::semantics::ModeChecking> modes{
      merkurius::semantics::CheckModes(
          tree, tokens, semanticForm, *resolution, *checking, messages)};
  merkurius::semantics::CheckDeterminism(tokens, *checking, *modes, messages);
  return result;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(
    const std::uint8_t *data, std::size_t size) {
  llvm::StringRef input{reinterpret_cast<const char *>(data), size};
  for (bool openMP : {false, true}) {
    Unparsed first{ConvertAndUnparse(input, openMP)};
    if (first.exact != input) {
      Fail("the parse tree does not unparse to the input", input, first.exact);
    }
    if (first.ok) {
      Unparsed canonical{ConvertAndUnparse(first.canonical, openMP)};
      if (!canonical.ok || canonical.tree != first.tree) {
        Fail("the canonical layout has a different parse tree", input,
            first.canonical);
      }
      if (canonical.canonical != first.canonical) {
        Fail("the canonical layout is not a fixed point", input,
            first.canonical + "\nbecame\n" + canonical.canonical);
      }
      Unparsed parenthesized{
          ConvertAndUnparse(first.parenthesized, openMP)};
      if (!parenthesized.ok || parenthesized.tree != first.tree) {
        Fail("the parenthesized form has a different parse tree", input,
            first.parenthesized);
      }
    }
  }
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
