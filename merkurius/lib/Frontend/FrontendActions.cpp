//===-- lib/Frontend/FrontendActions.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Frontend/FrontendActions.h"
#include "merkurius/Driver/Options.h"
#include "merkurius/Frontend/CompilerInstance.h"
#include "merkurius/Parser/convert.h"
#include "merkurius/Parser/lexer.h"
#include "merkurius/Parser/message.h"
#include "merkurius/Parser/provenance.h"
#include "merkurius/Parser/source.h"
#include "merkurius/Parser/term-reader.h"
#include "merkurius/Parser/unparse.h"
#include "merkurius/Semantics/desugar.h"
#include "merkurius/Semantics/module-loader.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"
#include <string>
#include <vector>

namespace merkurius::frontend {

bool FrontendAction::run(CompilerInstance &ci, llvm::StringRef input) {
  parser::AllSources &allSources{ci.getAllSources()};
  std::string error;
  llvm::raw_string_ostream errorStream{error};
  parser::SourceFile *sourceFile{input == "-"
          ? allSources.ReadStandardInput(errorStream)
          : allSources.OpenPath(input.str(), errorStream)};
  if (!sourceFile) {
    options::reportError("cannot open input file '" + input + "': " + error);
    return false;
  }
  instance_ = &ci;
  sourceFile_ = sourceFile;
  bool ok{executeAction()};
  sourceFile_ = nullptr;
  instance_ = nullptr;
  return ok;
}

bool PrintPreprocessedInputAction::executeAction() {
  llvm::raw_ostream *os{getInstance().createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  if (getSourceFile().hasByteOrderMark()) {
    *os << "\xef\xbb\xbf";
  }
  *os << getSourceFile().content();
  os->flush();
  return true;
}

bool DebugDumpProvenanceAction::executeAction() {
  llvm::raw_ostream *os{getInstance().createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  getInstance().getAllSources().Dump(*os);
  os->flush();
  return true;
}

bool LexAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  parser::LexerOptions options;
  options.openMP = ci.getLangOpts().openMP;
  parser::Messages messages;
  parser::TokenSequence tokens{parser::Lex(getSourceFile(), options, messages)};
  messages.Emit(llvm::errs(), ci.getAllSources());
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  executeLexAction(*os, tokens);
  os->flush();
  return !messages.AnyFatalError();
}

void DebugDumpTokensAction::executeLexAction(
    llvm::raw_ostream &os, const parser::TokenSequence &tokens) {
  parser::DumpTokens(os, tokens, getSourceFile());
}

void DebugUnparseTokensAction::executeLexAction(
    llvm::raw_ostream &os, const parser::TokenSequence &tokens) {
  parser::UnparseTokens(os, tokens);
}

bool DebugDumpTermsAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  parser::LexerOptions options;
  options.openMP = ci.getLangOpts().openMP;
  parser::Messages messages;
  parser::TokenSequence tokens{parser::Lex(getSourceFile(), options, messages)};
  // The lexer's messages, by location, to find the items they fall in.
  std::vector<const char *> lexerErrors;
  for (const parser::Message &message : messages.messages()) {
    if (const parser::CharBlock * at{message.GetCharBlock()}) {
      lexerErrors.push_back(at->begin());
    }
  }
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  parser::TermArena arena;
  parser::TermReader reader{tokens, arena, messages};
  std::size_t readerMessages{messages.messages().size()};
  while (std::optional<parser::ItemTerm> item{reader.ReadItem()}) {
    const parser::Token &first{tokens.tokens[item->tokens.first]};
    const parser::Token &last{tokens.tokens[item->tokens.last]};
    bool lexerError{llvm::any_of(lexerErrors, [&](const char *at) {
      return at >= first.spelling.begin() && at <= last.spelling.end();
    })};
    bool readerError{messages.messages().size() != readerMessages};
    readerMessages = messages.messages().size();
    if (item->term && !lexerError && !readerError) {
      parser::PrintCanonical(*os, *item->term);
    } else {
      *os << "error";
    }
    *os << '\n';
    arena.Reset();
  }
  os->flush();
  messages.Emit(llvm::errs(), ci.getAllSources());
  return !messages.AnyFatalError();
}

bool DebugDumpParseTreeNoSemaAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  parser::LexerOptions options;
  options.openMP = ci.getLangOpts().openMP;
  parser::Messages messages;
  parser::TokenSequence tokens{parser::Lex(getSourceFile(), options, messages)};
  parser::ParseTreeArena arena;
  const parser::ParseTree &tree{parser::ConvertFile(tokens, arena, messages)};
  messages.Emit(llvm::errs(), ci.getAllSources());
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  parser::DumpParseTree(*os, tree);
  os->flush();
  return !messages.AnyFatalError();
}

bool DebugUnparseExactAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  parser::LexerOptions options;
  options.openMP = ci.getLangOpts().openMP;
  parser::Messages messages;
  parser::TokenSequence tokens{parser::Lex(getSourceFile(), options, messages)};
  parser::ParseTreeArena arena;
  const parser::ParseTree &tree{parser::ConvertFile(tokens, arena, messages)};
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  // An item left out for an error leaves tokens that no node accounts for.
  bool complete{!messages.AnyFatalError()};
  parser::UnparseExact(*os, tree, tokens,
      minimalLayout_ ? parser::UnparseLayout::Minimal
                     : parser::UnparseLayout::Exact,
      complete, messages);
  os->flush();
  messages.Emit(llvm::errs(), ci.getAllSources());
  return !messages.AnyFatalError();
}

bool DebugUnparseNoSemaAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  parser::LexerOptions options;
  options.openMP = ci.getLangOpts().openMP;
  parser::Messages messages;
  parser::TokenSequence tokens{parser::Lex(getSourceFile(), options, messages)};
  parser::ParseTreeArena arena;
  const parser::ParseTree &tree{parser::ConvertFile(tokens, arena, messages)};
  messages.Emit(llvm::errs(), ci.getAllSources());
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  parser::UnparseCanonical(*os, tree, tokens, parenthesize_);
  os->flush();
  return !messages.AnyFatalError();
}

bool DebugDumpSemanticFormAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  parser::LexerOptions options;
  options.openMP = ci.getLangOpts().openMP;
  parser::Messages messages;
  parser::TokenSequence tokens{parser::Lex(getSourceFile(), options, messages)};
  parser::ParseTreeArena arena;
  const parser::ParseTree &tree{parser::ConvertFile(tokens, arena, messages)};
  semantics::SemanticArena semanticArena;
  const semantics::SemanticForm &form{
      semantics::Desugar(tree, tokens, semanticArena, messages)};
  messages.Emit(llvm::errs(), ci.getAllSources());
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  semantics::DumpSemanticForm(*os, form, ci.getFrontendOpts().dumpNonlocals);
  os->flush();
  return !messages.AnyFatalError();
}

bool DebugDumpImportsAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  for (const std::string &dir : ci.getFrontendOpts().searchDirectories) {
    ci.getAllSources().AppendSearchPathDirectory(dir);
  }
  parser::LexerOptions options;
  options.openMP = ci.getLangOpts().openMP;
  parser::Messages messages;
  parser::TokenSequence tokens{parser::Lex(getSourceFile(), options, messages)};
  parser::ParseTreeArena arena;
  const parser::ParseTree &tree{parser::ConvertFile(tokens, arena, messages)};
  semantics::ModuleLoader loader{ci.getAllSources(),
      ci.getFrontendOpts().libraryDirectory, messages};
  llvm::StringRef directory{
      llvm::sys::path::parent_path(getSourceFile().path())};
  semantics::ModuleLoader::Imports imports{
      loader.LoadImports(tree, tokens, directory.empty() ? "." : directory)};
  messages.Emit(llvm::errs(), ci.getAllSources());
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  auto names{[&](llvm::StringRef label,
                 const std::vector<const semantics::ModuleInterface *> &list) {
    if (!list.empty()) {
      *os << ' ' << label;
      for (const semantics::ModuleInterface *module : list) {
        *os << ' ' << module->name;
      }
    }
  }};
  *os << "imports:";
  names("import_module", imports.imports);
  names("use_module", imports.uses);
  *os << '\n';
  // Each module by its file's name, not its path, which depends on where
  // the library is.
  for (const semantics::ModuleInterface *module : loader.modules()) {
    *os << module->name << " ("
        << llvm::sys::path::filename(module->file->path()) << ", "
        << module->items.size() << " items):";
    names("import_module", module->imports);
    names("use_module", module->uses);
    *os << '\n';
  }
  os->flush();
  return !messages.AnyFatalError();
}

} // namespace merkurius::frontend
