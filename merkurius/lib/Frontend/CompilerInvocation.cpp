//===-- lib/Frontend/CompilerInvocation.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Frontend/CompilerInvocation.h"
#include "merkurius/Driver/Options.h"
#include "llvm/ADT/StringExtras.h"

using namespace merkurius::options;
using llvm::opt::InputArgList;

namespace merkurius::frontend {

static bool parseFrontendArgs(FrontendOptions &opts, const InputArgList &args) {
  opts.showHelp = args.hasArg(OPT_help);
  opts.showVersion = args.hasArg(OPT_version);
  // As in Flang, the last action option wins.
  if (const llvm::opt::Arg *arg =
          args.getLastArg(OPT_E, OPT_fdebug_dump_provenance,
              OPT_fdebug_dump_tokens, OPT_fdebug_unparse_tokens,
              OPT_fdebug_dump_terms, OPT_fdebug_dump_parse_tree_no_sema,
              OPT_fdebug_unparse_exact,
              OPT_fdebug_unparse_exact_minimal_layout,
              OPT_fdebug_unparse_no_sema,
              OPT_fdebug_unparse_parenthesized_no_sema,
              OPT_fdebug_dump_semantic_form, OPT_fdebug_dump_nonlocals,
              OPT_fdebug_dump_imports, OPT_fdebug_dump_resolved_names,
              OPT_fdebug_dump_types, OPT_fdebug_dump_modes,
              OPT_fdebug_dump_determinism, OPT_fdebug_dump_polymorphism,
              OPT_emit_merk, OPT_emit_mlir, OPT_emit_llvm, OPT_S,
              OPT_emit_obj)) {
    switch (arg->getOption().getID()) {
    case OPT_E:
      opts.programAction = ActionKind::PrintPreprocessedInput;
      break;
    case OPT_fdebug_dump_provenance:
      opts.programAction = ActionKind::DebugDumpProvenance;
      break;
    case OPT_fdebug_dump_tokens:
      opts.programAction = ActionKind::DebugDumpTokens;
      break;
    case OPT_fdebug_unparse_tokens:
      opts.programAction = ActionKind::DebugUnparseTokens;
      break;
    case OPT_fdebug_dump_terms:
      opts.programAction = ActionKind::DebugDumpTerms;
      break;
    case OPT_fdebug_dump_parse_tree_no_sema:
      opts.programAction = ActionKind::DebugDumpParseTreeNoSema;
      break;
    case OPT_fdebug_unparse_exact:
      opts.programAction = ActionKind::DebugUnparseExact;
      break;
    case OPT_fdebug_unparse_exact_minimal_layout:
      opts.programAction = ActionKind::DebugUnparseExactMinimalLayout;
      break;
    case OPT_fdebug_unparse_no_sema:
      opts.programAction = ActionKind::DebugUnparseNoSema;
      break;
    case OPT_fdebug_unparse_parenthesized_no_sema:
      opts.programAction = ActionKind::DebugUnparseParenthesizedNoSema;
      break;
    case OPT_fdebug_dump_semantic_form:
      opts.programAction = ActionKind::DebugDumpSemanticForm;
      break;
    case OPT_fdebug_dump_types:
      opts.programAction = ActionKind::DebugDumpTypes;
      break;
    case OPT_fdebug_dump_modes:
      opts.programAction = ActionKind::DebugDumpModes;
      break;
    case OPT_fdebug_dump_determinism:
      opts.programAction = ActionKind::DebugDumpDeterminism;
      break;
    case OPT_fdebug_dump_polymorphism:
      opts.programAction = ActionKind::DebugDumpPolymorphism;
      break;
    case OPT_emit_merk:
      opts.programAction = ActionKind::EmitMerk;
      break;
    case OPT_emit_mlir:
      opts.programAction = ActionKind::EmitMLIR;
      break;
    case OPT_emit_llvm:
      opts.programAction = ActionKind::EmitLLVM;
      break;
    case OPT_S:
      opts.programAction = ActionKind::EmitAssembly;
      break;
    case OPT_emit_obj:
      opts.programAction = ActionKind::EmitObj;
      break;
    case OPT_fdebug_dump_resolved_names:
      opts.programAction = ActionKind::DebugDumpResolvedNames;
      break;
    case OPT_fdebug_dump_imports:
      opts.programAction = ActionKind::DebugDumpImports;
      break;
    case OPT_fdebug_dump_nonlocals:
      opts.programAction = ActionKind::DebugDumpSemanticForm;
      opts.dumpNonlocals = true;
      break;
    }
  }
  opts.outputFile = args.getLastArgValue(OPT_o).str();
  opts.searchDirectories = args.getAllArgValues(OPT_I);
  opts.libraryDirectory = args.getLastArgValue(OPT_library_dir).str();
  opts.inferTypes = !args.hasArg(OPT_fno_infer_types);
  if (const llvm::opt::Arg *arg =
          args.getLastArg(OPT_ftype_inference_iteration_limit_EQ)) {
    llvm::StringRef value{arg->getValue()};
    if (value.getAsInteger(10, opts.typeInferenceIterationLimit) ||
        opts.typeInferenceIterationLimit == 0) {
      reportError(
          "invalid value '" + value + "' in '" + arg->getAsString(args) + "'");
      return false;
    }
  }
  opts.inferModes = args.hasArg(OPT_finfer_modes);
  opts.mmcUniqueness = args.hasArg(OPT_fmmc_uniqueness);
  if (const llvm::opt::Arg *arg =
          args.getLastArg(OPT_fmode_inference_iteration_limit_EQ)) {
    llvm::StringRef value{arg->getValue()};
    if (value.getAsInteger(10, opts.modeInferenceIterationLimit) ||
        opts.modeInferenceIterationLimit == 0) {
      reportError(
          "invalid value '" + value + "' in '" + arg->getAsString(args) + "'");
      return false;
    }
  }
  opts.inputs = args.getAllArgValues(OPT_INPUT);
  if (opts.inputs.empty() && !opts.showHelp && !opts.showVersion) {
    reportError("no input files");
    return false;
  }
  return true;
}

static bool parseLangArgs(LangOptions &opts, const InputArgList &args) {
  opts.openMP = args.hasArg(OPT_fopenmp);
  if (const llvm::opt::Arg *arg = args.getLastArg(OPT_fopenmp_version_EQ)) {
    llvm::StringRef value{arg->getValue()};
    if (value.getAsInteger(10, opts.openMPVersion)) {
      reportError(
          "invalid value '" + value + "' in '" + arg->getAsString(args) + "'");
      return false;
    }
  }
  return true;
}

static bool parseCodeGenArgs(CodeGenOptions &opts, const InputArgList &args) {
  opts.triple = args.getLastArgValue(OPT_triple).str();
  opts.cpu = args.getLastArgValue(OPT_target_cpu).str();
  opts.features = llvm::join(args.getAllArgValues(OPT_target_feature), ",");
  if (const llvm::opt::Arg *arg = args.getLastArg(OPT_O)) {
    llvm::StringRef value{arg->getValue()};
    if (value.empty()) {
      opts.optimizationLevel = 1; // -O, as in Clang
    } else if (value.getAsInteger(10, opts.optimizationLevel) ||
        opts.optimizationLevel > 3) {
      reportError(
          "invalid value '" + value + "' in '" + arg->getAsString(args) + "'");
      return false;
    }
  }
  return true;
}

bool CompilerInvocation::createFromArgs(
    CompilerInvocation &res, llvm::ArrayRef<const char *> args) {
  std::optional<InputArgList> parsed{parseArgs(args, FC1Option)};
  if (!parsed) {
    return false;
  }
  bool ok{parseFrontendArgs(res.getFrontendOpts(), *parsed)};
  ok &= parseLangArgs(res.getLangOpts(), *parsed);
  ok &= parseCodeGenArgs(res.getCodeGenOpts(), *parsed);
  return ok;
}

} // namespace merkurius::frontend
