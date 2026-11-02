//===-- include/merkurius/Frontend/FrontendOptions.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_FRONTEND_FRONTENDOPTIONS_H_
#define MERKURIUS_FRONTEND_FRONTENDOPTIONS_H_

#include "merkurius/CodeGen/TargetOptions.h"
#include <string>
#include <vector>

namespace merkurius::frontend {

// The action -fc1 runs on each input.
enum class ActionKind {
  // -E: print the source as the lexer reads it. There is no prescanner that
  // rewrites it, so this is the input unchanged, byte order mark included.
  PrintPreprocessedInput,
  // -fdebug-dump-provenance: dump the provenance table of the sources.
  DebugDumpProvenance,
  // -fdebug-dump-tokens: dump the tokens, one per line.
  DebugDumpTokens,
  // -fdebug-unparse-tokens: print every token's trivia and spelling, which
  // gives back the source file byte for byte.
  DebugUnparseTokens,
  // -fdebug-dump-terms: print each item's term in canonical notation, one
  // per line, or "error" for an item with a diagnostic.
  DebugDumpTerms,
  DebugDumpParseTreeNoSema,
  // -fdebug-unparse-exact: print the parse tree from the tokens it recorded,
  // which gives back the source file byte for byte, and report any token
  // the tree does not account for.
  DebugUnparseExact,
  // -fdebug-unparse-exact-minimal-layout: the same, without comments and
  // line number directives, and with each token's trivia reduced to one
  // newline or space.
  DebugUnparseExactMinimalLayout,
  // -fdebug-unparse-no-sema: print the parse tree in the canonical layout.
  DebugUnparseNoSema,
  // -fdebug-unparse-parenthesized-no-sema: the same, with every operator
  // term in parentheses.
  DebugUnparseParenthesizedNoSema,
  // -fdebug-dump-semantic-form: desugar the clauses and print their
  // semantic form; -fdebug-dump-nonlocals also prints each goal's nonlocal
  // variables.
  DebugDumpSemanticForm,
  // -fdebug-dump-imports: load the imported modules and list their
  // interfaces.
  DebugDumpImports,
  // -fdebug-dump-resolved-names: resolve the names and list the module's
  // symbols and each name's candidates.
  DebugDumpResolvedNames,
  // -fdebug-dump-types: type check, and list each clause's variable types
  // and chosen declarations.
  DebugDumpTypes,
  // -fdebug-dump-modes: type, purity and mode check, and list each
  // predicate's and function's procedures.
  DebugDumpModes,
  // -fdebug-dump-determinism: type, purity and mode check, and list each
  // procedure's body, with its switches.
  DebugDumpDeterminism,
  // -fdebug-dump-polymorphism: check the source file as for
  // -fdebug-dump-determinism, and list each procedure's type_info
  // arguments and those its goals pass.
  DebugDumpPolymorphism,
  // -emit-merk: generate the module's merk operations and write them.
  EmitMerk,
  // -emit-mlir: the same, lowered to MLIR's LLVM dialect.
  EmitMLIR,
  // -emit-llvm: the same, translated to LLVM IR and optimized.
  EmitLLVM,
  // -S: the target's assembly for it.
  EmitAssembly,
  // -emit-obj: its object file.
  EmitObj,
};

struct FrontendOptions {
  ActionKind programAction{ActionKind::PrintPreprocessedInput};
  std::vector<std::string> inputs;
  // Empty or "-" means standard output.
  std::string outputFile;
  bool showHelp{false};
  bool showVersion{false};
  bool dumpNonlocals{false}; // -fdebug-dump-nonlocals
  std::vector<std::string> searchDirectories; // -I
  // -library-dir, or lib/merkurius beside the executable's directory.
  std::string libraryDirectory;
  bool inferTypes{true}; // -fno-infer-types turns it off
  unsigned typeInferenceIterationLimit{60}; // -ftype-inference-iteration-limit=
  bool inferModes{false}; // -finfer-modes
  unsigned modeInferenceIterationLimit{30}; // -fmode-inference-iteration-limit=
  bool mmcUniqueness{false}; // -fmmc-uniqueness
};

// The options of code generation: the target, and how far to optimize.
using CodeGenOptions = codegen::TargetOptions;

struct LangOptions {
  // -fopenmp: read %$omp directive lines instead of skipping them as
  // comments.
  bool openMP{false};
  // -fopenmp-version=, as in Clang and Flang: 52 for OpenMP 5.2.
  unsigned openMPVersion{52};
};

} // namespace merkurius::frontend

#endif // MERKURIUS_FRONTEND_FRONTENDOPTIONS_H_
