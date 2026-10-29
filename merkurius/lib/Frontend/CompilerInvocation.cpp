//===-- lib/Frontend/CompilerInvocation.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Frontend/CompilerInvocation.h"
#include "merkurius/Driver/Options.h"

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
              OPT_fdebug_dump_terms)) {
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
    }
  }
  opts.outputFile = args.getLastArgValue(OPT_o).str();
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

bool CompilerInvocation::createFromArgs(
    CompilerInvocation &res, llvm::ArrayRef<const char *> args) {
  std::optional<InputArgList> parsed{parseArgs(args, FC1Option)};
  if (!parsed) {
    return false;
  }
  bool ok{parseFrontendArgs(res.getFrontendOpts(), *parsed)};
  ok &= parseLangArgs(res.getLangOpts(), *parsed);
  return ok;
}

} // namespace merkurius::frontend
