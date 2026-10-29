//===-- tools/merkc/driver.cpp - Merkurius compiler driver ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The entry point of merkc. With -fc1 as its first argument it runs the
// frontend; otherwise it is the driver, which works out the jobs for the
// command line and runs each of them as a separate merkc -fc1 process.
//
//===----------------------------------------------------------------------===//

#include "merkurius/Driver/Options.h"
#include "merkurius/Version.inc"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

using namespace merkurius::options;
using llvm::opt::InputArgList;

int fc1_main(llvm::ArrayRef<const char *> argv);

namespace {
// One command to run: the program followed by its arguments.
using Job = std::vector<std::string>;
} // namespace

static std::string getExecutablePath(const char *argv0) {
  // Any address in this program will do.
  void *mainAddr{reinterpret_cast<void *>(&getExecutablePath)};
  return llvm::sys::fs::getMainExecutable(argv0, mainAddr);
}

static void printVersion(llvm::raw_ostream &os, llvm::StringRef executable) {
  os << "merkc version " MERKURIUS_VERSION_STRING "\n";
  os << "InstalledDir: " << llvm::sys::path::parent_path(executable) << '\n';
}

static void printJob(llvm::raw_ostream &os, const Job &job) {
  for (const std::string &arg : job) {
    os << ' ';
    llvm::sys::printArg(os, arg, /*Quote=*/true);
  }
  os << '\n';
}

static bool isMercurySource(llvm::StringRef input) {
  return input == "-" || llvm::sys::path::extension(input) == ".m";
}

// Builds one -fc1 job per input, or returns false after reporting an error.
static bool buildJobs(const InputArgList &args, llvm::StringRef executable,
    std::vector<Job> &jobs) {
  std::vector<std::string> inputs{args.getAllArgValues(OPT_INPUT)};
  if (inputs.empty()) {
    reportError("no input files");
    return false;
  }
  if (!args.hasArg(OPT_E)) {
    reportError("only -E is implemented so far");
    return false;
  }
  llvm::StringRef output{args.getLastArgValue(OPT_o)};
  if (!output.empty() && inputs.size() > 1) {
    reportError("cannot specify -o when generating multiple output files");
    return false;
  }
  bool ok{true};
  for (const std::string &input : inputs) {
    if (!isMercurySource(input)) {
      reportError("'" + input + "' is not a Mercury source file (.m)");
      ok = false;
      continue;
    }
    Job job{executable.str(), "-fc1", "-E"};
    if (args.hasArg(OPT_fopenmp)) {
      job.push_back("-fopenmp");
    }
    if (const llvm::opt::Arg *arg = args.getLastArg(OPT_fopenmp_version_EQ)) {
      job.push_back(arg->getAsString(args));
    }
    if (!output.empty()) {
      job.push_back("-o");
      job.push_back(output.str());
    }
    job.push_back(input);
    jobs.push_back(std::move(job));
  }
  return ok;
}

static int runJob(const Job &job) {
  llvm::SmallVector<llvm::StringRef> argv(job.begin(), job.end());
  std::string errMsg;
  int rc{llvm::sys::ExecuteAndWait(argv[0], argv, /*Env=*/std::nullopt,
      /*Redirects=*/{}, /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg)};
  if (rc < 0) {
    reportError("unable to execute '" + job[0] + "': " + errMsg);
  }
  return rc;
}

int main(int argc, const char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::ArrayRef<const char *> args(argv, argc);

  if (args.size() >= 2 && llvm::StringRef(args[1]) == "-fc1") {
    return fc1_main(args.drop_front(2));
  }

  std::optional<InputArgList> parsed{
      parseArgs(args.drop_front(), DriverOption)};
  if (!parsed) {
    return 1;
  }
  std::string executable{getExecutablePath(args[0])};

  if (parsed->hasArg(OPT_help)) {
    getDriverOptTable().printHelp(llvm::outs(), "merkc [options] file...",
        "Merkurius compiler", /*ShowHidden=*/false, /*ShowAllAliases=*/false,
        llvm::opt::Visibility(DriverOption));
    return 0;
  }
  if (parsed->hasArg(OPT_version)) {
    printVersion(llvm::outs(), executable);
    return 0;
  }

  bool printOnly{parsed->hasArg(OPT__HASH_HASH_HASH)};
  bool verbose{parsed->hasArg(OPT_v)};
  if (printOnly || verbose) {
    printVersion(llvm::errs(), executable);
    if (!parsed->hasArg(OPT_INPUT)) {
      return 0;
    }
  }

  std::vector<Job> jobs;
  if (!buildJobs(*parsed, executable, jobs)) {
    return 1;
  }
  for (const Job &job : jobs) {
    if (printOnly || verbose) {
      printJob(llvm::errs(), job);
    }
    if (!printOnly && runJob(job) != 0) {
      return 1;
    }
  }
  return 0;
}
