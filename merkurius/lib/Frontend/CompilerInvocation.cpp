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
  if (args.hasArg(OPT_E)) {
    opts.programAction = ActionKind::PrintPreprocessedInput;
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
