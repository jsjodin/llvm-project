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

static bool isObjectFile(llvm::StringRef input) {
  return llvm::sys::path::extension(input) == ".o";
}

// The options the driver passes on to each -fc1 job.
static void addFrontendOptions(const InputArgList &args, Job &job) {
  if (args.hasArg(OPT_fopenmp)) {
    job.push_back("-fopenmp");
  }
  if (const llvm::opt::Arg *arg = args.getLastArg(OPT_fopenmp_version_EQ)) {
    job.push_back(arg->getAsString(args));
  }
  for (const std::string &dir : args.getAllArgValues(OPT_I)) {
    job.push_back("-I");
    job.push_back(dir);
  }
  if (const llvm::opt::Arg *arg = args.getLastArg(OPT_target_EQ)) {
    job.push_back("-triple");
    job.push_back(arg->getValue());
  }
  if (const llvm::opt::Arg *arg = args.getLastArg(OPT_O)) {
    job.push_back(arg->getAsString(args));
  }
}

// The program that links: Clang, beside merkc or else on the PATH (design
// document, "Linking and offloading").
static std::string findLinker(llvm::StringRef executable) {
  llvm::SmallString<128> beside{llvm::sys::path::parent_path(executable)};
  llvm::sys::path::append(beside, "clang");
  if (llvm::sys::fs::can_execute(beside)) {
    return beside.str().str();
  }
  if (llvm::ErrorOr<std::string> found{llvm::sys::findProgramByName("clang")}) {
    return *found;
  }
  return "clang";
}

// The jobs for the command line, or false after reporting an error: with
// -E, -S or -c, one -fc1 job per source; otherwise one per source, writing
// a temporary object file, and a link of the objects with the runtime
// library. `temporaries` gets the temporary files' names.
static bool buildJobs(const InputArgList &args, llvm::StringRef executable,
    std::vector<Job> &jobs, std::vector<std::string> &temporaries) {
  std::vector<std::string> inputs{args.getAllArgValues(OPT_INPUT)};
  if (inputs.empty()) {
    reportError("no input files");
    return false;
  }
  bool preprocess{args.hasArg(OPT_E)};
  bool assemble{!preprocess && args.hasArg(OPT_S)};
  bool compile{!preprocess && !assemble && args.hasArg(OPT_c)};
  bool link{!preprocess && !assemble && !compile};
  llvm::StringRef output{args.getLastArgValue(OPT_o)};
  unsigned sources{0};
  for (const std::string &input : inputs) {
    sources += isMercurySource(input);
  }
  if (!output.empty() && !link && sources > 1) {
    reportError("cannot specify -o when generating multiple output files");
    return false;
  }
  bool ok{true};
  std::vector<std::string> objects;
  for (const std::string &input : inputs) {
    if (link && isObjectFile(input)) {
      objects.push_back(input);
      continue;
    }
    if (!isMercurySource(input)) {
      reportError("'" + input + "' is not a Mercury source file (.m)");
      ok = false;
      continue;
    }
    Job job{executable.str(), "-fc1"};
    if (preprocess) {
      job.push_back("-E");
    } else if (assemble) {
      job.push_back("-S");
    } else {
      job.push_back("-emit-obj");
    }
    addFrontendOptions(args, job);
    std::string jobOutput{link ? "" : output.str()};
    if (link) {
      llvm::SmallString<128> temporary;
      llvm::sys::fs::getPotentiallyUniqueTempFileName(
          llvm::sys::path::stem(input), "o", temporary);
      jobOutput = temporary.str().str();
      temporaries.push_back(jobOutput);
      objects.push_back(jobOutput);
    }
    if (!jobOutput.empty()) {
      job.push_back("-o");
      job.push_back(jobOutput);
    }
    job.push_back(input);
    jobs.push_back(std::move(job));
  }
  if (ok && link) {
    // The runtime library is in lib beside bin/merkc, in the build
    // directory and the installation alike.
    llvm::SmallString<128> libraryDir{llvm::sys::path::parent_path(
        llvm::sys::path::parent_path(executable))};
    llvm::sys::path::append(libraryDir, "lib");
    Job job{findLinker(executable)};
    if (const llvm::opt::Arg *arg = args.getLastArg(OPT_target_EQ)) {
      job.push_back(arg->getAsString(args));
    }
    job.push_back("-o");
    job.push_back(output.empty() ? "a.out" : output.str());
    job.insert(job.end(), objects.begin(), objects.end());
    job.push_back("-L" + libraryDir.str().str());
    job.push_back("-lmerkrt");
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
  std::vector<std::string> temporaries;
  if (!buildJobs(*parsed, executable, jobs, temporaries)) {
    return 1;
  }
  int result{0};
  for (const Job &job : jobs) {
    if (printOnly || verbose) {
      printJob(llvm::errs(), job);
    }
    if (!printOnly && runJob(job) != 0) {
      result = 1;
      break;
    }
  }
  for (const std::string &temporary : temporaries) {
    llvm::sys::fs::remove(temporary);
  }
  return result;
}
