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
#include "merkurius/Semantics/check-modes.h"
#include "merkurius/Semantics/check-purity.h"
#include "merkurius/Semantics/check-types.h"
#include "merkurius/Semantics/desugar.h"
#include "merkurius/Semantics/module-loader.h"
#include "merkurius/Semantics/resolve-names.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"
#include <functional>
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

namespace {
// The front end up to name resolution, for the actions that need it.
struct Resolved {
  parser::Messages messages;
  parser::TokenSequence tokens;
  parser::ParseTreeArena arena;
  const parser::ParseTree *tree{nullptr};
  semantics::SemanticArena semanticArena;
  const semantics::SemanticForm *form{nullptr};
  std::unique_ptr<semantics::ModuleLoader> loader;
  std::unique_ptr<semantics::NameResolution> resolution;
};

void Resolve(CompilerInstance &ci, parser::SourceFile &file,
    Resolved &r) {
  for (const std::string &dir : ci.getFrontendOpts().searchDirectories) {
    ci.getAllSources().AppendSearchPathDirectory(dir);
  }
  parser::LexerOptions options;
  options.openMP = ci.getLangOpts().openMP;
  r.tokens = parser::Lex(file, options, r.messages);
  r.tree = &parser::ConvertFile(r.tokens, r.arena, r.messages);
  r.form = &semantics::Desugar(*r.tree, r.tokens, r.semanticArena, r.messages);
  r.loader = std::make_unique<semantics::ModuleLoader>(ci.getAllSources(),
      ci.getFrontendOpts().libraryDirectory, r.messages);
  llvm::StringRef directory{llvm::sys::path::parent_path(file.path())};
  r.loader->LoadImports(
      *r.tree, r.tokens, directory.empty() ? "." : directory);
  r.resolution = semantics::ResolveNames(
      *r.tree, r.tokens, *r.form, *r.loader, r.messages);
}

// A position in the source as line:column.
std::string Position(const parser::AllSources &sources, parser::CharBlock at) {
  if (auto range{sources.GetProvenanceRange(at)}) {
    if (auto pos{sources.GetSourcePosition(range->start())}) {
      return std::to_string(pos->line) + ':' + std::to_string(pos->column);
    }
  }
  return "";
}

// The same, after the file's name, for a position in another module.
std::string FilePosition(const parser::AllSources &sources, parser::CharBlock at) {
  if (auto range{sources.GetProvenanceRange(at)}) {
    if (auto pos{sources.GetSourcePosition(range->start())}) {
      return llvm::sys::path::filename(*pos->path).str() + ':' +
          std::to_string(pos->line) + ':' + std::to_string(pos->column);
    }
  }
  return "";
}

parser::CharBlock Source(
    const parser::TokenSequence &tokens, parser::TokenRange range) {
  return {tokens.tokens[range.first].spelling.begin(),
      tokens.tokens[range.last].spelling.end()};
}
} // namespace

bool DebugDumpResolvedNamesAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  Resolved r;
  Resolve(ci, getSourceFile(), r);
  r.messages.Emit(llvm::errs(), ci.getAllSources());
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  const parser::AllSources &sources{ci.getAllSources()};
  const semantics::NameResolution &resolution{*r.resolution};
  *os << "symbols of " << resolution.module().name() << ":\n";
  for (const semantics::Symbol *symbol : resolution.module().symbols()) {
    *os << "  " << Position(sources, symbol->source) << ' ' << *symbol
        << (symbol->exported ? " exported" : "");
    if (symbol->owner) {
      *os << " of " << *symbol->owner;
    }
    *os << '\n';
  }
  *os << "names:\n";
  for (const semantics::ResolvedName &name : resolution.names()) {
    parser::CharBlock text{Source(r.tokens, name.name->tokens)};
    *os << "  " << Position(sources, text) << ' ' << text.ToString() << ' '
        << semantics::EnumToString(name.use) << '/' << name.arity << " ->";
    if (name.candidates.empty()) {
      *os << (name.use == semantics::NameUse::DataTerm &&
                     semantics::IsCharConstant(*name.name, name.arity)
                 ? " char constant"
                 : " undefined");
    }
    llvm::interleave(
        name.candidates, *os,
        [&](const semantics::Symbol *symbol) { *os << ' ' << *symbol; },
        ",");
    *os << '\n';
  }
  os->flush();
  return !r.messages.AnyFatalError();
}

bool DebugDumpTypesAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  Resolved r;
  Resolve(ci, getSourceFile(), r);
  semantics::TypeCheckOptions options;
  options.inferTypes = ci.getFrontendOpts().inferTypes;
  options.inferenceIterationLimit =
      ci.getFrontendOpts().typeInferenceIterationLimit;
  std::unique_ptr<semantics::TypeChecking> checking{semantics::CheckTypes(
      *r.form, *r.resolution, r.tokens, r.messages, options)};
  semantics::CheckPurity(
      *r.tree, r.tokens, *r.form, *r.resolution, *checking, r.messages);
  r.messages.Emit(llvm::errs(), ci.getAllSources());
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  const parser::AllSources &sources{ci.getAllSources()};
  for (const semantics::ClauseTypes &clause : checking->clauses()) {
    const semantics::ClauseForm &form{*clause.clause};
    *os << "clause at " << Position(sources, Source(r.tokens, form.tokens));
    if (clause.definition) {
      *os << " of " << *clause.definition;
    }
    if (clause.method) {
      *os << " in instance " << clause.method->instanceText;
    }
    if (const semantics::Signature *sig{clause.method ? &clause.method->signature
                : clause.definition ? checking->SignatureOf(*clause.definition)
                                    : nullptr}) {
      *os << (clause.inferred ? ": inferred " : ": ")
          << semantics::SignatureToString(*clause.definition, *sig);
    }
    *os << '\n';
    for (semantics::VarId var{0}; var < form.vars.size(); ++var) {
      const semantics::VarInfo &info{form.vars[var]};
      if (info.kind == semantics::VarKind::Program) {
        *os << "  " << info.name;
      } else if (info.kind == semantics::VarKind::State) {
        *os << "  " << info.name << '!' << info.version;
      } else {
        continue;
      }
      *os << ": " << checking->Print(clause, *clause.vars[var]) << '\n';
    }
    for (unsigned goal{0}; goal < form.goals; ++goal) {
      if (const parser::SymName *name{clause.names[goal]}) {
        parser::CharBlock text{Source(r.tokens, name->tokens)};
        *os << "  " << Position(sources, text) << ' ' << text.ToString()
            << " -> ";
        if (const semantics::Symbol *symbol{clause.symbols[goal]}) {
          *os << *symbol;
        } else {
          *os << "char constant";
        }
        *os << '\n';
      }
    }
    // Each constraint of a goal, with how it is satisfied, and below it
    // those of the instance that satisfies it.
    std::function<void(const semantics::ConstraintProof &, unsigned)> proof{
        [&](const semantics::ConstraintProof &p, unsigned depth) {
          *os << std::string(depth, ' ') << checking->Print(clause, p.constraint);
          if (p.instance) {
            const parser::TokenSequence &tokens{*p.instance->tokens};
            parser::CharBlock at{
                Source(tokens, p.instance->decl->name->tokens)};
            *os << " by the instance at "
                << (p.instance->local ? Position(sources, at)
                                      : FilePosition(sources, at));
          } else if (p.goal) {
            const parser::SymName *name{clause.names[*p.goal]};
            *os << " by the existential constraint of "
                << (name ? Source(r.tokens, name->tokens).ToString() : "?")
                << " at "
                << (name ? Position(sources, Source(r.tokens, name->tokens))
                         : std::string{"?"});
          } else if (p.assumed) {
            *os << " by the declaration's "
                << checking->Print(clause, *p.assumed);
          }
          *os << '\n';
          for (const semantics::ConstraintProof &premise : p.premises) {
            proof(premise, depth + 2);
          }
        }};
    for (const semantics::GoalConstraint &c : clause.constraints) {
      if (c.goal >= form.goals) {
        *os << "  existential constraint of the declaration: ";
        proof(c.proof, 0);
        continue;
      }
      const parser::SymName *name{clause.names[c.goal]};
      *os << "  constraint of "
          << (name ? Source(r.tokens, name->tokens).ToString() : "?")
          << " at "
          << (name ? Position(sources, Source(r.tokens, name->tokens))
                   : std::string{"?"})
          << ": ";
      proof(c.proof, 0);
    }
  }
  os->flush();
  return !r.messages.AnyFatalError();
}

bool DebugDumpModesAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  const FrontendOptions &frontend{ci.getFrontendOpts()};
  Resolved r;
  Resolve(ci, getSourceFile(), r);
  semantics::TypeCheckOptions typeOptions;
  typeOptions.inferTypes = frontend.inferTypes;
  typeOptions.inferenceIterationLimit = frontend.typeInferenceIterationLimit;
  std::unique_ptr<semantics::TypeChecking> types{semantics::CheckTypes(
      *r.form, *r.resolution, r.tokens, r.messages, typeOptions)};
  semantics::CheckPurity(
      *r.tree, r.tokens, *r.form, *r.resolution, *types, r.messages);
  semantics::ModeCheckOptions options;
  options.inferModes = frontend.inferModes;
  options.inferenceIterationLimit = frontend.modeInferenceIterationLimit;
  std::unique_ptr<semantics::ModeChecking> modes{semantics::CheckModes(*r.tree,
      r.tokens, *r.form, *r.resolution, *types, r.messages, options)};
  r.messages.Emit(llvm::errs(), ci.getAllSources());
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  const parser::AllSources &sources{ci.getAllSources()};
  for (const semantics::Symbol *symbol : modes->instDefinitions()) {
    *os << Position(sources, symbol->source) << ' ' << *symbol << " == ";
    if (const semantics::Inst *body{modes->insts().Body(*symbol)}) {
      semantics::PrintInst(*os, *body);
    } else {
      *os << "(none)";
    }
    *os << '\n';
  }
  for (const semantics::Procedure &procedure : modes->procedures()) {
    *os << (procedure.source.empty() ? std::string{"(default)"}
                                     : Position(sources, procedure.source))
        << ' ' << *procedure.symbol << " mode " << procedure.number << ": "
        << procedure.symbol->name;
    bool function{procedure.symbol->kind == semantics::SymbolKind::Function};
    llvm::ArrayRef<semantics::ArgMode> arguments{procedure.modes};
    if (function && !arguments.empty()) {
      arguments = arguments.drop_back();
    }
    if (!arguments.empty()) {
      *os << '(';
      llvm::interleave(
          arguments, *os,
          [&](const semantics::ArgMode &mode) {
            semantics::PrintArgMode(*os, mode);
          },
          ", ");
      *os << ')';
    }
    if (function && !procedure.modes.empty()) {
      *os << " = ";
      semantics::PrintArgMode(*os, procedure.modes.back());
    }
    if (procedure.determinism) {
      *os << " is " << semantics::DeterminismName(*procedure.determinism);
    }
    *os << '\n';
    if (modes->HasError(procedure)) {
      *os << "  mode error\n";
    } else if (semantics::Unchecked why{modes->UncheckedReason(procedure)};
               why != semantics::Unchecked::None) {
      *os << "  not checked: " << semantics::EnumToString(why) << '\n';
    }
    for (const semantics::ClauseModes &clause : modes->ClausesOf(procedure)) {
      *os << "  clause at "
          << Position(sources, Source(r.tokens, clause.clause->clause->tokens))
          << '\n';
      semantics::DumpClauseModes(*os, clause, r.tokens,
          [&](parser::CharBlock at) { return Position(sources, at); });
    }
  }
  os->flush();
  return !r.messages.AnyFatalError();
}

} // namespace merkurius::frontend
