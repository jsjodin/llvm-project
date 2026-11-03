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
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include <string>
#include <vector>

using namespace merkurius::options;
using llvm::opt::InputArgList;

int fc1_main(llvm::ArrayRef<const char *> argv);

namespace {
// One command to run: the program followed by its arguments.
using Job = std::vector<std::string>;
} // namespace

// The triple the host's offload entries are for, as Flang's driver gives it
// with --offload-targets=; each GPU's device triple adds its subarchitecture.
static constexpr llvm::StringLiteral kOffloadTargetAMDGPU{"amdgpu-amd-amdhsa"};

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

// The options the driver passes on to each -fc1 job: for the host, or with
// `deviceTriple` for an OpenMP device.
static void addFrontendOptions(const InputArgList &args, Job &job,
    llvm::StringRef deviceTriple = "") {
  if (args.hasArg(OPT_fopenmp)) {
    job.push_back("-fopenmp");
  }
  if (const llvm::opt::Arg *arg = args.getLastArg(OPT_fopenmp_version_EQ)) {
    job.push_back(arg->getAsString(args));
  }
  if (args.hasArg(OPT_fdo_loops)) {
    job.push_back("-fdo-loops");
  }
  for (const std::string &dir : args.getAllArgValues(OPT_I)) {
    job.push_back("-I");
    job.push_back(dir);
  }
  if (!deviceTriple.empty()) {
    job.push_back("-triple");
    job.push_back(deviceTriple.str());
  } else if (const llvm::opt::Arg *arg = args.getLastArg(OPT_target_EQ)) {
    job.push_back("-triple");
    job.push_back(arg->getValue());
  }
  if (const llvm::opt::Arg *arg = args.getLastArg(OPT_O)) {
    job.push_back(arg->getAsString(args));
  }
}

namespace {
// A GPU of --offload-arch=, and the triple its device code is compiled for.
struct OffloadArch {
  std::string arch;
  std::string triple;
};
} // namespace

// The GPUs that --offload-arch= names, or false after reporting an error.
// Each AMD GPU's triple has its subarchitecture, as Clang's and Flang's
// drivers give it: gfx1030's is amdgpu10.30-amd-amdhsa.
static bool getOffloadArchs(
    const InputArgList &args, std::vector<OffloadArch> &archs) {
  std::vector<std::string> values{args.getAllArgValues(OPT_offload_arch_EQ)};
  if (!values.empty() && !args.hasArg(OPT_fopenmp)) {
    reportError("--offload-arch needs -fopenmp");
    return false;
  }
  for (const std::string &value : values) {
    llvm::AMDGPU::GPUKind kind{llvm::AMDGPU::parseArchAMDGCN(value)};
    if (kind == llvm::AMDGPU::GK_NONE) {
      reportError("offloading to '" + value +
          "' is not supported yet: only AMD GPUs, such as gfx1030, are");
      return false;
    }
    llvm::Triple triple{kOffloadTargetAMDGPU};
    triple.setArch(triple.getArch(), llvm::AMDGPU::getSubArch(kind));
    archs.push_back({value, triple.str()});
  }
  return true;
}

// A temporary file's name, from `stem` and `suffix`, added to `temporaries`.
static std::string temporaryFile(llvm::StringRef stem, llvm::StringRef suffix,
    std::vector<std::string> &temporaries) {
  llvm::SmallString<128> name;
  llvm::sys::fs::getPotentiallyUniqueTempFileName(stem, suffix, name);
  temporaries.push_back(name.str().str());
  return temporaries.back();
}

// The jobs that compile `input` to the object file `object` with its target
// regions offloaded to `archs`, as Flang's are: the host's bitcode, with its
// offload entries; each GPU's, which matches them; the GPUs' code in one
// offload binary; and the host's object file, from its bitcode, with the
// offload binary embedded, for the linker wrapper to extract.
static void addOffloadJobs(const InputArgList &args, llvm::StringRef executable,
    const std::string &input, const std::string &object,
    const std::vector<OffloadArch> &archs, std::vector<Job> &jobs,
    std::vector<std::string> &temporaries) {
  llvm::StringRef stem{llvm::sys::path::stem(input)};
  std::string offloadTargets{"--offload-targets=" +
      std::string{kOffloadTargetAMDGPU}};
  std::string hostBitcode{temporaryFile(stem, "bc", temporaries)};
  Job host{executable.str(), "-fc1", "-emit-llvm-bc"};
  addFrontendOptions(args, host);
  host.insert(host.end(), {offloadTargets, "-o", hostBitcode, input});
  jobs.push_back(std::move(host));
  Job binary{findTool(executable, "llvm-offload-binary")};
  if (binary[0].empty()) {
    binary[0] = "llvm-offload-binary";
  }
  std::string offloadBinary{temporaryFile(stem, "out", temporaries)};
  binary.insert(binary.end(), {"-o", offloadBinary});
  for (const OffloadArch &arch : archs) {
    std::string deviceBitcode{
        temporaryFile((stem + "-" + arch.arch).str(), "bc", temporaries)};
    Job device{executable.str(), "-fc1", "-emit-llvm-bc"};
    addFrontendOptions(args, device, arch.triple);
    device.insert(device.end(),
        {"-fopenmp-is-target-device", "-fopenmp-host-ir-file-path",
            hostBitcode, "-o", deviceBitcode, input});
    jobs.push_back(std::move(device));
    binary.push_back("--image=file=" + deviceBitcode + ",triple=" +
        arch.triple + ",arch=" + arch.arch + ",kind=openmp");
  }
  jobs.push_back(std::move(binary));
  Job embed{executable.str(), "-fc1", "-emit-obj"};
  addFrontendOptions(args, embed);
  embed.insert(embed.end(),
      {"-fembed-offload-object=" + offloadBinary, offloadTargets, "-o", object,
          "-x", "ir", hostBitcode});
  jobs.push_back(std::move(embed));
}

// The program that links: Clang, beside merkc or else on the PATH (design
// document, "Linking and offloading").
static std::string findLinker(llvm::StringRef executable) {
  std::string clang{findClang(executable)};
  return clang.empty() ? "clang" : clang;
}

// The jobs for the command line, or false after reporting an error: with
// -E, -S or -c, one -fc1 job per source; otherwise one per source, writing
// a temporary object file, and a link of the objects with the runtime
// library. With --offload-arch=, each source's object file comes from the
// offload jobs instead. `temporaries` gets the temporary files' names.
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
  std::vector<OffloadArch> archs;
  if (!getOffloadArchs(args, archs)) {
    return false;
  }
  bool offload{!archs.empty() && !preprocess};
  if (offload && assemble) {
    reportError("-S with --offload-arch is not supported yet");
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
    if (offload) {
      std::string object{link ? temporaryFile(llvm::sys::path::stem(input),
                                    "o", temporaries)
              : !output.empty()
              ? output.str()
              : (llvm::sys::path::stem(input) + ".o").str()};
      if (link) {
        objects.push_back(object);
      }
      addOffloadJobs(
          args, executable, input, object, archs, jobs, temporaries);
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
    // The compiled standard library and the runtime library are in lib
    // beside bin/merkc, in the build directory and the installation alike;
    // the standard library's foreign code may call C's maths library.
    llvm::SmallString<128> libraryDir{llvm::sys::path::parent_path(
        llvm::sys::path::parent_path(executable))};
    llvm::sys::path::append(libraryDir, "lib");
    Job job{findLinker(executable)};
    // -v shows the link's own jobs too, such as the linker wrapper's for
    // the GPUs' code, as Flang's -v does.
    if (args.hasArg(OPT_v)) {
      job.push_back("-v");
    }
    if (const llvm::opt::Arg *arg = args.getLastArg(OPT_target_EQ)) {
      job.push_back(arg->getAsString(args));
    }
    // Offloading links as Clang does with -fopenmp and --offload-arch=: it
    // runs the linker wrapper, which links the GPUs' code, and adds the
    // OpenMP runtime libraries.
    if (!archs.empty()) {
      job.push_back("-fopenmp");
      std::string offloadArch{"--offload-arch="};
      for (const OffloadArch &arch : archs) {
        offloadArch += (&arch == &archs.front() ? "" : ",") + arch.arch;
      }
      job.push_back(offloadArch);
    }
    job.push_back("-o");
    job.push_back(output.empty() ? "a.out" : output.str());
    job.insert(job.end(), objects.begin(), objects.end());
    for (const std::string &dir : args.getAllArgValues(OPT_L)) {
      job.push_back("-L" + dir);
    }
    job.push_back("-L" + libraryDir.str().str());
    job.push_back("-lmerkstd");
    job.push_back("-lmerkrt");
    job.push_back("-lm");
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
