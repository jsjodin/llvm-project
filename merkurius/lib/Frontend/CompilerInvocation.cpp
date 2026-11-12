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
#include "llvm/TargetParser/Triple.h"

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
              OPT_fdebug_dump_types, OPT_fdebug_dump_loops,
              OPT_fdebug_dump_modes,
              OPT_fdebug_dump_determinism, OPT_fdebug_dump_polymorphism,
              OPT_emit_merk, OPT_emit_merk_bc, OPT_emit_mlir, OPT_emit_llvm, OPT_emit_llvm_bc, OPT_S,
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
    case OPT_fdebug_dump_loops:
      opts.programAction = ActionKind::DebugDumpLoops;
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
    case OPT_emit_merk_bc:
      opts.programAction = ActionKind::EmitMerkBytecode;
      break;
    case OPT_emit_mlir:
      opts.programAction = ActionKind::EmitMLIR;
      break;
    case OPT_emit_llvm:
      opts.programAction = ActionKind::EmitLLVM;
      break;
    case OPT_emit_llvm_bc:
      opts.programAction = ActionKind::EmitLLVMBitcode;
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
  if (const llvm::opt::Arg *arg = args.getLastArg(OPT_x)) {
    llvm::StringRef value{arg->getValue()};
    if (value == "mercury") {
      opts.inputLanguage = InputLanguage::Mercury;
    } else if (value == "mlir") {
      opts.inputLanguage = InputLanguage::MLIR;
    } else if (value == "ir") {
      opts.inputLanguage = InputLanguage::LLVMIR;
    } else {
      reportError("invalid value '" + value + "' in '-x " + value + "'");
      return false;
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
  opts.stubUnimplemented = args.hasArg(OPT_fstub_unimplemented);
  opts.warnUnselectedMessage = args.hasFlag(
      OPT_Wunselected_message, OPT_Wno_unselected_message, true);
  opts.traceFlags = args.getAllArgValues(OPT_trace_flag_EQ);
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
  // GPU processes' kernels read the directives of the loops they run across
  // teams, which their compile makes without OpenMP (-fprocess-loops).
  opts.openMP = args.hasArg(OPT_fopenmp) || args.hasArg(OPT_fprocess_loops);
  opts.doLoops = args.hasArg(OPT_fdo_loops);
  opts.receiveGoal = args.hasArg(OPT_freceive_goal);
  opts.processDeclarations = args.hasArg(OPT_fprocess_declarations);
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
  opts.openMP = args.hasArg(OPT_fopenmp);
  if (const llvm::opt::Arg *arg = args.getLastArg(OPT_fopenmp_version_EQ)) {
    llvm::StringRef{arg->getValue()}.getAsInteger(10, opts.openMPVersion);
  }
  opts.openMPIsTargetDevice = args.hasArg(OPT_fopenmp_is_target_device);
  opts.openMPHostIRFile =
      args.getLastArgValue(OPT_fopenmp_host_ir_file_path).str();
  opts.openMPTargets = args.getAllArgValues(OPT_offload_targets_EQ);
  opts.offloadObjects = args.getAllArgValues(OPT_fembed_offload_object_EQ);
  opts.processKernels = args.hasArg(OPT_fprocess_kernels);
  opts.processImages = args.getAllArgValues(OPT_fembed_process_image_EQ);
  if (opts.processKernels && opts.openMP) {
    reportError("-fprocess-kernels compiles without OpenMP: it cannot be "
                "given with -fopenmp");
    return false;
  }
  if (args.hasArg(OPT_fprocess_loops) && !opts.processKernels) {
    reportError("-fprocess-loops is given only with -fprocess-kernels");
    return false;
  }
  if (opts.processKernels && !llvm::Triple{opts.triple}.isAMDGPU()) {
    reportError("-fprocess-kernels needs the triple of an AMD GPU, such as "
                "amdgpu10.30-amd-amdhsa");
    return false;
  }
  if ((opts.openMPIsTargetDevice || !opts.openMPTargets.empty()) &&
      !opts.openMP) {
    reportError("OpenMP offloading needs -fopenmp");
    return false;
  }
  if (const llvm::opt::Arg *arg = args.getLastArg(OPT_O)) {
    opts.optimizationLevelGiven = true;
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
