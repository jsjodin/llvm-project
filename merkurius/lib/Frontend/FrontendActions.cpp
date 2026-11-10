//===-- lib/Frontend/FrontendActions.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Frontend/FrontendActions.h"
#include "merkurius/CodeGen/Lowering.h"
#include "merkurius/CodeGen/MLIRGen.h"
#include "merkurius/Driver/Options.h"
#include "llvm/Support/FileSystem.h"
#include "merkurius/Frontend/CompilerInstance.h"
#include "merkurius/Parser/convert.h"
#include "merkurius/Parser/lexer.h"
#include "merkurius/Parser/message.h"
#include "merkurius/Parser/provenance.h"
#include "merkurius/Parser/source.h"
#include "merkurius/Parser/term-reader.h"
#include "merkurius/Parser/unparse.h"
#include "merkurius/Semantics/check-determinism.h"
#include "merkurius/Semantics/check-uniqueness.h"
#include "merkurius/Semantics/check-modes.h"
#include "merkurius/Semantics/check-purity.h"
#include "merkurius/Semantics/check-receives.h"
#include "merkurius/Semantics/check-types.h"
#include "merkurius/Semantics/desugar.h"
#include "merkurius/Semantics/module-loader.h"
#include "merkurius/Semantics/polymorphism.h"
#include "merkurius/Semantics/resolve-names.h"
#include "merkurius/Semantics/translate-loops.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "merkurius/Dialect/Mrt/IR/MrtDialect.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"
#include <string>
#include <vector>

namespace merkurius::frontend {

static parser::ReaderOptions ReaderOptionsOf(CompilerInstance &ci) {
  parser::ReaderOptions options;
  options.doLoops = ci.getLangOpts().doLoops;
  options.receiveGoal = ci.getLangOpts().receiveGoal;
  return options;
}

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
  parser::TermReader reader{tokens, arena, messages,
      parser::TermReader::defaultMaxDepth, ReaderOptionsOf(ci)};
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
  const parser::ParseTree &tree{parser::ConvertFile(tokens, arena, messages,
      parser::TermReader::defaultMaxDepth, ReaderOptionsOf(ci))};
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
  const parser::ParseTree &tree{parser::ConvertFile(tokens, arena, messages,
      parser::TermReader::defaultMaxDepth, ReaderOptionsOf(ci))};
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
  const parser::ParseTree &tree{parser::ConvertFile(tokens, arena, messages,
      parser::TermReader::defaultMaxDepth, ReaderOptionsOf(ci))};
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
  const parser::ParseTree &tree{parser::ConvertFile(tokens, arena, messages,
      parser::TermReader::defaultMaxDepth, ReaderOptionsOf(ci))};
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

// The modules of a list, after its label, if there are any.
static void PrintModuleNames(llvm::raw_ostream &os, llvm::StringRef label,
    const std::vector<const semantics::ModuleInterface *> &list) {
  if (!list.empty()) {
    os << ' ' << label;
    for (const semantics::ModuleInterface *module : list) {
      os << ' ' << module->name;
    }
  }
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
  const parser::ParseTree &tree{parser::ConvertFile(tokens, arena, messages,
      parser::TermReader::defaultMaxDepth, ReaderOptionsOf(ci))};
  semantics::ModuleLoader loader{ci.getAllSources(),
      ci.getFrontendOpts().libraryDirectory, messages,
      !ci.getFrontendOpts().mmcUniqueness};
  llvm::StringRef directory{
      llvm::sys::path::parent_path(getSourceFile().path())};
  semantics::ModuleLoader::Imports imports{
      loader.LoadImports(tree, tokens, directory.empty() ? "." : directory)};
  messages.Emit(llvm::errs(), ci.getAllSources());
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  *os << "imports:";
  PrintModuleNames(*os, "import_module", imports.imports);
  PrintModuleNames(*os, "use_module", imports.uses);
  *os << '\n';
  // Each module by its file's name, not its path, which depends on where
  // the library is.
  for (const semantics::ModuleInterface *module : loader.modules()) {
    *os << module->name << " ("
        << llvm::sys::path::filename(module->file->path()) << ", "
        << module->items.size() << " items):";
    PrintModuleNames(*os, "import_module", module->imports);
    PrintModuleNames(*os, "use_module", module->uses);
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
  r.tree = &parser::ConvertFile(r.tokens, r.arena, r.messages,
      parser::TermReader::defaultMaxDepth, ReaderOptionsOf(ci));
  r.form = &semantics::Desugar(*r.tree, r.tokens, r.semanticArena, r.messages);
  r.loader = std::make_unique<semantics::ModuleLoader>(ci.getAllSources(),
      ci.getFrontendOpts().libraryDirectory, r.messages,
      !ci.getFrontendOpts().mmcUniqueness);
  llvm::StringRef directory{llvm::sys::path::parent_path(file.path())};
  r.loader->LoadImports(
      *r.tree, r.tokens, directory.empty() ? "." : directory);
  // The modules that loops' translations call, which need not be imported.
  for (llvm::StringRef module : semantics::LoopModules(*r.form)) {
    r.loader->Load(module, r.tokens.tokens.front().spelling,
        directory.empty() ? "." : directory);
  }
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

namespace {
// A constraint of a clause, with how it is satisfied, and below it, indented,
// those of the instance that satisfies it.
struct ProofPrinter {
  void Print(const semantics::ConstraintProof &p, unsigned depth) const {
    os << std::string(depth, ' ') << checking.Print(clause, p.constraint);
    if (p.instance) {
      const parser::TokenSequence &tokens{*p.instance->tokens};
      parser::CharBlock at{Source(tokens, p.instance->decl->name->tokens)};
      os << " by the instance at "
         << (p.instance->local ? Position(sources, at)
                               : FilePosition(sources, at));
    } else if (p.goal) {
      const parser::SymName *name{clause.names[*p.goal]};
      os << " by the existential constraint of "
         << (name ? Source(clauseTokens, name->tokens).ToString() : "?")
         << " at "
         << (name ? Position(sources, Source(clauseTokens, name->tokens))
                  : std::string{"?"});
    } else if (p.assumed) {
      os << " by the declaration's " << checking.Print(clause, *p.assumed);
    }
    os << '\n';
    for (const semantics::ConstraintProof &premise : p.premises) {
      Print(premise, depth + 2);
    }
  }

  llvm::raw_ostream &os;
  const semantics::TypeChecking &checking;
  const semantics::ClauseTypes &clause;
  const parser::AllSources &sources;
  const parser::TokenSequence &clauseTokens; // the clause's module's
};
} // namespace

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
    ProofPrinter proof{*os, *checking, clause, sources, r.tokens};
    for (const semantics::GoalConstraint &c : clause.constraints) {
      if (c.goal >= form.goals) {
        *os << "  existential constraint of the declaration: ";
        proof.Print(c.proof, 0);
        continue;
      }
      const parser::SymName *name{clause.names[c.goal]};
      *os << "  constraint of "
          << (name ? Source(r.tokens, name->tokens).ToString() : "?")
          << " at "
          << (name ? Position(sources, Source(r.tokens, name->tokens))
                   : std::string{"?"})
          << ": ";
      proof.Print(c.proof, 0);
    }
  }
  os->flush();
  return !r.messages.AnyFatalError();
}

// A procedure as -fdebug-dump-modes lists it: pred m.p/2 mode 0: p(in,
// out) is det, with a rejected inferred mode's final insts not_reached.
static void PrintProcedure(llvm::raw_ostream &os,
    const semantics::Procedure &procedure, semantics::ModeChecking &modes,
    const parser::AllSources &sources) {
  os << (procedure.source.empty() ? std::string{"(default)"}
                                  : Position(sources, procedure.source))
     << ' ' << *procedure.symbol
     << (procedure.inferred ? " inferred mode " : " mode ") << procedure.number
     << ": " << procedure.symbol->name;
  bool function{procedure.symbol->kind == semantics::SymbolKind::Function};
  llvm::ArrayRef<semantics::ArgMode> arguments{procedure.modes};
  // A rejected inferred mode's final insts are not known, as in mmc.
  std::vector<semantics::ArgMode> rejected;
  if (procedure.inferred && modes.HasError(procedure)) {
    for (const semantics::ArgMode &mode : procedure.modes) {
      rejected.push_back({mode.initial, modes.insts().NotReached()});
    }
    arguments = rejected;
  }
  if (function && !arguments.empty()) {
    arguments = arguments.drop_back();
  }
  if (!arguments.empty()) {
    os << '(';
    llvm::interleave(
        arguments, os,
        [&](const semantics::ArgMode &mode) { semantics::PrintArgMode(os, mode); },
        ", ");
    os << ')';
  }
  if (function && !procedure.modes.empty()) {
    os << " = ";
    semantics::PrintArgMode(os, procedure.modes.back());
  }
  if (procedure.determinism) {
    os << " is " << semantics::DeterminismName(*procedure.determinism);
  }
  os << '\n';
}

bool DebugDumpLoopsAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  const FrontendOptions &frontend{ci.getFrontendOpts()};
  Resolved r;
  Resolve(ci, getSourceFile(), r);
  semantics::TypeCheckOptions typeOptions;
  typeOptions.inferTypes = frontend.inferTypes;
  typeOptions.inferenceIterationLimit = frontend.typeInferenceIterationLimit;
  std::unique_ptr<semantics::TypeChecking> types{semantics::CheckTypes(
      *r.form, *r.resolution, r.tokens, r.messages, typeOptions)};
  semantics::LoopPurities loops;
  semantics::CheckPurity(
      *r.tree, r.tokens, *r.form, *r.resolution, *types, r.messages, &loops);
  semantics::LoopTranslation translation{semantics::TranslateLoops(
      *r.resolution, *types, loops, r.tokens, r.semanticArena, r.messages)};
  r.messages.Emit(llvm::errs(), ci.getAllSources());
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  const parser::AllSources &sources{ci.getAllSources()};
  for (std::size_t i : translation.clauses) {
    const semantics::ClauseTypes &clause{types->clauses()[i]};
    *os << Position(sources, Source(r.tokens, clause.clause->tokens)) << ' ';
    if (const semantics::Signature *sig{clause.method
                ? &clause.method->signature
                : types->SignatureOf(*clause.definition)}) {
      *os << semantics::SignatureToString(*clause.definition, *sig);
    } else {
      *os << *clause.definition;
    }
    *os << '\n';
    const semantics::ClauseForm *form{clause.clause};
    semantics::DumpSemanticForm(*os,
        semantics::SemanticForm{llvm::ArrayRef<const semantics::ClauseForm *>{form}},
        true);
  }
  os->flush();
  return !r.messages.AnyFatalError();
}

// A procedure's modes, whether they were checked, and each clause's.
static void DumpProcedureModes(llvm::raw_ostream &os,
    const semantics::Procedure &procedure, semantics::ModeChecking &modes,
    const parser::AllSources &sources, const parser::TokenSequence &tokens) {
  PrintProcedure(os, procedure, modes, sources);
  if (modes.HasError(procedure)) {
    os << "  mode error\n";
  } else if (semantics::Unchecked why{modes.UncheckedReason(procedure)};
             why != semantics::Unchecked::None) {
    os << "  not checked: " << semantics::EnumToString(why) << '\n';
  }
  for (const semantics::ClauseModes &clause : modes.ClausesOf(procedure)) {
    os << "  clause at "
       << Position(sources, Source(tokens, clause.clause->clause->tokens))
       << '\n';
    semantics::DumpClauseModes(os, clause, tokens,
        [&](parser::CharBlock at) { return Position(sources, at); });
  }
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
  semantics::LoopPurities loops;
  semantics::CheckPurity(
      *r.tree, r.tokens, *r.form, *r.resolution, *types, r.messages, &loops);
  semantics::TranslateLoops(
      *r.resolution, *types, loops, r.tokens, r.semanticArena, r.messages);
  semantics::ModeCheckOptions options;
  options.inferModes = frontend.inferModes;
  options.inferenceIterationLimit = frontend.modeInferenceIterationLimit;
  options.mmcUniqueness = frontend.mmcUniqueness;
  options.stubUnimplemented = frontend.stubUnimplemented;
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
  // The declared procedures, and then those inferred from calls:
  // pred m.p/2 inferred mode 0: p(in, uo).
  for (const semantics::Procedure &procedure : modes->procedures()) {
    DumpProcedureModes(*os, procedure, *modes, sources, r.tokens);
  }
  // By predicate, as in the source, and then in the order made.
  std::vector<const semantics::Procedure *> inferred;
  for (const semantics::Procedure &procedure : modes->inferred()) {
    inferred.push_back(&procedure);
  }
  std::stable_sort(inferred.begin(), inferred.end(),
      [](const semantics::Procedure *a, const semantics::Procedure *b) {
        return a->symbol->source.begin() < b->symbol->source.begin();
      });
  for (const semantics::Procedure *procedure : inferred) {
    DumpProcedureModes(*os, *procedure, *modes, sources, r.tokens);
  }
  // Instance methods: method m.c(T) pred m.p/1 mode 0, and their clauses.
  for (const semantics::MethodModes &method : modes->methods()) {
    *os << "method " << method.method->instanceText << ' '
        << *method.procedure->symbol << " mode " << method.procedure->number
        << '\n';
    if (method.error) {
      *os << "  mode error\n";
    } else if (method.unchecked != semantics::Unchecked::None) {
      *os << "  not checked: " << semantics::EnumToString(method.unchecked)
          << '\n';
    }
    for (const semantics::ClauseModes &clause : method.clauses) {
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

namespace {

// The passes up to uniqueness checking, which work on the bodies that
// determinism analysis builds.
struct Analysed {
  std::unique_ptr<semantics::TypeChecking> types;
  std::unique_ptr<semantics::ModeChecking> modes;
  std::unique_ptr<semantics::DeterminismChecking> determinism;
  semantics::LoopTranslation loops;
};

Analysed Analyse(CompilerInstance &ci, Resolved &r) {
  const FrontendOptions &frontend{ci.getFrontendOpts()};
  semantics::TypeCheckOptions typeOptions;
  typeOptions.inferTypes = frontend.inferTypes;
  typeOptions.inferenceIterationLimit = frontend.typeInferenceIterationLimit;
  std::unique_ptr<semantics::TypeChecking> types{semantics::CheckTypes(
      *r.form, *r.resolution, r.tokens, r.messages, typeOptions)};
  semantics::LoopPurities loops;
  semantics::CheckPurity(
      *r.tree, r.tokens, *r.form, *r.resolution, *types, r.messages, &loops);
  semantics::LoopTranslation translation{semantics::TranslateLoops(
      *r.resolution, *types, loops, r.tokens, r.semanticArena, r.messages)};
  semantics::ModeCheckOptions options;
  options.inferModes = frontend.inferModes;
  options.inferenceIterationLimit = frontend.modeInferenceIterationLimit;
  options.mmcUniqueness = frontend.mmcUniqueness;
  options.stubUnimplemented = frontend.stubUnimplemented;
  bool earlier{r.messages.AnyFatalError()};
  std::unique_ptr<semantics::ModeChecking> modes{semantics::CheckModes(*r.tree,
      r.tokens, *r.form, *r.resolution, *types, r.messages, options)};
  if (frontend.warnUnselectedMessage) {
    semantics::CheckReceives(r.tokens, *types, modes->constructors(), r.messages);
  }
  std::unique_ptr<semantics::DeterminismChecking> determinism{
      semantics::CheckDeterminism(
          r.tokens, *types, *modes, r.messages, !earlier)};
  semantics::CheckUniqueness(
      r.tokens, *modes, *determinism, r.messages, frontend.mmcUniqueness);
  return {std::move(types), std::move(modes), std::move(determinism),
      std::move(translation)};
}

// Prints a body's procedure as -fdebug-dump-modes lists it, with the
// determinism inferred for it; an instance method's as method m.c(T)
// pred m.p/1 mode 0.
void PrintBodyHeader(llvm::raw_ostream &os,
    const semantics::ProcedureBody &body, semantics::ModeChecking &modes,
    const parser::AllSources &sources) {
  if (body.method) {
    os << "method " << body.method->method->instanceText << ' '
       << *body.procedure->symbol << " mode " << body.procedure->number
       << '\n';
    return;
  }
  std::string header;
  llvm::raw_string_ostream line{header};
  PrintProcedure(line, *body.procedure, modes, sources);
  os << llvm::StringRef{header}.rtrim('\n');
  if (body.determinism) {
    os << ", inferred " << semantics::DeterminismName(*body.determinism);
  }
  os << '\n';
}

} // namespace

bool DebugDumpDeterminismAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  Resolved r;
  Resolve(ci, getSourceFile(), r);
  Analysed a{Analyse(ci, r)};
  r.messages.Emit(llvm::errs(), ci.getAllSources());
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  const parser::AllSources &sources{ci.getAllSources()};
  // Each procedure whose modes were checked, and its body.
  for (const semantics::ProcedureBody &body : a.determinism->bodies()) {
    PrintBodyHeader(*os, body, *a.modes, sources);
    semantics::DumpProcedureBody(*os, body, r.tokens,
        [&](parser::CharBlock at) { return Position(sources, at); });
  }
  os->flush();
  return !r.messages.AnyFatalError();
}

bool DebugDumpPolymorphismAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  Resolved r;
  Resolve(ci, getSourceFile(), r);
  Analysed a{Analyse(ci, r)};
  std::unique_ptr<semantics::PolymorphismAnalysis> polymorphism{
      semantics::AnalysePolymorphism(*a.types, *a.modes, *a.determinism)};
  r.messages.Emit(llvm::errs(), ci.getAllSources());
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  const parser::AllSources &sources{ci.getAllSources()};
  // Each procedure whose modes were checked, its type_info arguments, and
  // the type_infos of its goals.
  for (const semantics::ProcedurePolymorphism &procedure :
      polymorphism->procedures()) {
    PrintBodyHeader(*os, *procedure.body, *a.modes, sources);
    semantics::DumpProcedurePolymorphism(*os, procedure, r.tokens,
        [&](parser::CharBlock at) { return Position(sources, at); });
  }
  os->flush();
  return !r.messages.AnyFatalError();
}

// An MLIR module from a .mlir file, in the merk dialect or lowered from it:
// its target is the one it records, as in a module written by -emit-merk or
// -emit-mlir, so that it lowers as it would have when written.
static mlir::OwningOpRef<mlir::ModuleOp> ReadMLIR(mlir::MLIRContext &context,
    const parser::SourceFile &file,
    std::unique_ptr<codegen::TargetInfo> &target) {
  context.loadDialect<merk::MerkDialect, mrt::MrtDialect, mlir::DLTIDialect,
      mlir::LLVM::LLVMDialect, mlir::omp::OpenMPDialect>();
  // Errors in it are reported at file:line:column, with the line.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(file.content(), file.path(),
          /*RequiresNullTerminator=*/false),
      llvm::SMLoc());
  mlir::SourceMgrDiagnosticHandler handler{sourceMgr, &context};
  mlir::ParserConfig config{&context};
  mlir::OwningOpRef<mlir::ModuleOp> module{
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, config)};
  if (!module) {
    return nullptr;
  }
  std::string error;
  target = codegen::TargetInfo::FromModule(*module, error);
  if (!target) {
    options::reportError(file.path() + ": " + error);
    return nullptr;
  }
  return module;
}

// Reports MLIR's diagnostics as the front end reports its messages, at
// file:line:column.
static void PrintDiagnostic(const mlir::Diagnostic &diagnostic) {
  std::string message;
  llvm::raw_string_ostream os{message};
  if (auto loc{mlir::dyn_cast<mlir::FileLineColLoc>(diagnostic.getLocation())}) {
    os << llvm::sys::path::filename(loc.getFilename().getValue()) << ':'
       << loc.getLine() << ':' << loc.getColumn() << ": ";
  }
  switch (diagnostic.getSeverity()) {
  case mlir::DiagnosticSeverity::Error:
    os << "error: ";
    break;
  case mlir::DiagnosticSeverity::Warning:
    os << "warning: ";
    break;
  default:
    os << "note: ";
    break;
  }
  os << diagnostic.str();
  llvm::errs() << message << '\n';
}

// An MLIR diagnostic, as file:line:column, then its notes, each so.
static mlir::LogicalResult ReportDiagnostic(mlir::Diagnostic &diagnostic) {
  PrintDiagnostic(diagnostic);
  for (const mlir::Diagnostic &note : diagnostic.getNotes()) {
    PrintDiagnostic(note);
  }
  return mlir::success();
}

// What the action's input is: as -x says, or else as its extension does.
static InputLanguage Language(
    InputLanguage language, const parser::SourceFile &file) {
  if (language != InputLanguage::FromExtension) {
    return language;
  }
  llvm::StringRef extension{llvm::sys::path::extension(file.path())};
  // A merk module, as -emit-merk or -emit-merk-bc writes it, or any other.
  if (extension == ".merk" || extension == ".merkbc" || extension == ".mlir") {
    return InputLanguage::MLIR;
  }
  if (extension == ".ll" || extension == ".bc") {
    return InputLanguage::LLVMIR;
  }
  return InputLanguage::Mercury;
}

bool CodeGenAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  mlir::MLIRContext context;
  mlir::ScopedDiagnosticHandler diagnostics{&context, ReportDiagnostic};
  // An error names the operation; printing it in generic form is for
  // debugging the compiler.
  context.printOpOnDiagnostic(false);
  std::unique_ptr<codegen::TargetInfo> target;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  switch (Language(ci.getFrontendOpts().inputLanguage, getSourceFile())) {
  case InputLanguage::MLIR:
    module = ReadMLIR(context, getSourceFile(), target);
    if (!module) {
      return false;
    }
    return Lower(*module, *target);
  case InputLanguage::LLVMIR:
    return LowerLLVMIR();
  case InputLanguage::FromExtension:
  case InputLanguage::Mercury:
    break;
  }
  Resolved r;
  Resolve(ci, getSourceFile(), r);
  Analysed a{Analyse(ci, r)};
  std::unique_ptr<semantics::PolymorphismAnalysis> polymorphism{
      semantics::AnalysePolymorphism(*a.types, *a.modes, *a.determinism)};
  if (r.messages.AnyFatalError()) {
    r.messages.Emit(llvm::errs(), ci.getAllSources());
    return false;
  }
  std::string error;
  target = codegen::TargetInfo::Create(ci.getCodeGenOpts(), error);
  if (!target) {
    r.messages.Emit(llvm::errs(), ci.getAllSources());
    options::reportError(error);
    return false;
  }
  // A device's compile copies the library's procedures from their merk
  // bytecode, which the library's build writes beside its sources.
  std::string deviceLibrary;
  if (target->options().openMPIsTargetDevice &&
      !ci.getFrontendOpts().libraryDirectory.empty()) {
    llvm::SmallString<256> path{ci.getFrontendOpts().libraryDirectory};
    llvm::sys::path::append(path, "device");
    deviceLibrary = path.str().str();
  }
  module = codegen::GenerateMerk(context, r.resolution->module().name(),
      *r.tree, r.tokens, ci.getAllSources(), *r.resolution, *a.types, *a.modes,
      *a.determinism, *polymorphism, a.loops, *target, r.messages,
      {ci.getFrontendOpts().stubUnimplemented,
          ci.getFrontendOpts().mmcUniqueness, ci.getFrontendOpts().traceFlags,
          deviceLibrary});
  r.messages.Emit(llvm::errs(), ci.getAllSources());
  if (!module || r.messages.AnyFatalError()) {
    return false;
  }
  if (mlir::failed(mlir::verify(*module))) {
    options::reportError("the generated merk module does not verify");
    return false;
  }
  return Lower(*module, *target);
}

llvm::raw_ostream *CodeGenAction::Output() {
  CompilerInstance &ci{getInstance()};
  // An object file or assembly is written beside the current directory's
  // other outputs, named after the source, as by Clang's -c and -S.
  std::string &outputFile{ci.getFrontendOpts().outputFile};
  if (outputFile.empty() &&
      (kind_ == ActionKind::EmitObj || kind_ == ActionKind::EmitAssembly)) {
    llvm::SmallString<128> name{llvm::sys::path::stem(getSourceFile().path())};
    name += kind_ == ActionKind::EmitObj ? ".o" : ".s";
    outputFile = name.str().str();
  }
  return ci.createDefaultOutputFile();
}

bool CodeGenAction::Lower(
    mlir::ModuleOp module, const codegen::TargetInfo &target) {
  std::string error;
  llvm::raw_ostream *os{Output()};
  if (!os) {
    return false;
  }
  if (kind_ == ActionKind::EmitMerk) {
    // Notes for a later pass, which only clutter what is read.
    module.walk([](merk::RootTypeOp op) { op.erase(); });
    // Printed without its roots' type_infos: it can be compiled again, for a
    // test, but what it makes cannot be collected.
    module->setAttr("merk.no_root_types", mlir::UnitAttr::get(module.getContext()));
    module.print(*os);
    return true;
  }
  if (kind_ == ActionKind::EmitMerkBytecode) {
    // Whole, with the notes that -emit-merk leaves out, as a module compiled
    // from it must be the same as one compiled from the source.
    if (mlir::failed(mlir::writeBytecodeToFile(module, *os))) {
      options::reportError("cannot write the merk module as bytecode");
      return false;
    }
    return true;
  }
  std::string clang{options::findClang(llvm::sys::fs::getMainExecutable(
      "merkc", reinterpret_cast<void *>(&options::findClang)))};
  if (mlir::failed(codegen::LowerToLLVMDialect(module, target, clang))) {
    return false;
  }
  if (kind_ == ActionKind::EmitMLIR) {
    module.print(*os);
    return true;
  }
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule{
      codegen::TranslateToLLVMIR(module, llvmContext, target)};
  if (!llvmModule) {
    options::reportError("cannot translate the module to LLVM IR");
    return false;
  }
  if (!codegen::LinkForeignCode(*llvmModule, module, target, clang, error)) {
    options::reportError(error);
    return false;
  }
  return Emit(*llvmModule, target, *os);
}

// LLVM IR read back, as Flang's -x ir is: for the host's offload compile,
// which embeds the device code in the host's IR from its first compile.
bool CodeGenAction::LowerLLVMIR() {
  CompilerInstance &ci{getInstance()};
  if (kind_ != ActionKind::EmitLLVM && kind_ != ActionKind::EmitLLVMBitcode &&
      kind_ != ActionKind::EmitAssembly && kind_ != ActionKind::EmitObj) {
    options::reportError("LLVM IR input can only be compiled to LLVM IR, "
                         "bitcode, assembly or an object file");
    return false;
  }
  std::string error;
  std::unique_ptr<codegen::TargetInfo> target{
      codegen::TargetInfo::Create(ci.getCodeGenOpts(), error)};
  if (!target) {
    options::reportError(error);
    return false;
  }
  llvm::LLVMContext llvmContext;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> llvmModule{
      llvm::parseIRFile(getSourceFile().path(), diagnostic, llvmContext)};
  if (!llvmModule) {
    diagnostic.print("merkc", llvm::errs());
    return false;
  }
  // The target is the compile's, as in Flang, which says when it is not
  // the module's.
  const llvm::Triple &triple{target->targetMachine().getTargetTriple()};
  if (llvmModule->getTargetTriple() != triple) {
    llvm::errs() << "merkc: warning: overriding the module target triple with "
                 << triple.str() << '\n';
  }
  llvmModule->setTargetTriple(triple);
  llvmModule->setDataLayout(target->dataLayout());
  llvm::raw_ostream *os{Output()};
  if (!os) {
    return false;
  }
  return Emit(*llvmModule, *target, *os);
}

bool CodeGenAction::Emit(llvm::Module &llvmModule,
    const codegen::TargetInfo &target, llvm::raw_ostream &os) {
  std::string error;
  if (!codegen::EmbedOffloadObjects(
          llvmModule, getInstance().getCodeGenOpts().offloadObjects, error)) {
    options::reportError(error);
    return false;
  }
  codegen::OptimizeLLVMIR(llvmModule, target);
  codegen::AddStackChecks(llvmModule);
  codegen::ExpandHeapAddresses(llvmModule);
  if (kind_ == ActionKind::EmitLLVM) {
    llvmModule.print(os, nullptr);
    return true;
  }
  if (kind_ == ActionKind::EmitLLVMBitcode) {
    llvm::WriteBitcodeToFile(llvmModule, os);
    return true;
  }
  llvm::SmallString<0> code;
  llvm::raw_svector_ostream codeStream{code};
  if (!codegen::EmitCode(llvmModule, target,
          kind_ == ActionKind::EmitObj ? llvm::CodeGenFileType::ObjectFile
                                       : llvm::CodeGenFileType::AssemblyFile,
          codeStream, error)) {
    options::reportError(error);
    return false;
  }
  os << code;
  os.flush();
  return true;
}

} // namespace merkurius::frontend
