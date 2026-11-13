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
// `deviceTriple` for a device, OpenMP's unless `openMP` is false, as for
// GPU processes' kernels, which are compiled without OpenMP.
static void addFrontendOptions(const InputArgList &args, Job &job,
    llvm::StringRef deviceTriple = "", bool openMP = true) {
  if (openMP && args.hasArg(OPT_fopenmp)) {
    job.push_back("-fopenmp");
  }
  if (const llvm::opt::Arg *arg = args.getLastArg(OPT_fopenmp_version_EQ);
      arg && openMP) {
    job.push_back(arg->getAsString(args));
  }
  if (args.hasArg(OPT_fdo_loops)) {
    job.push_back("-fdo-loops");
  }
  if (args.hasArg(OPT_freceive_goal)) {
    job.push_back("-freceive-goal");
  }
  if (args.hasArg(OPT_fprocess_declarations)) {
    job.push_back("-fprocess-declarations");
  }
  if (args.hasArg(OPT_fstub_unimplemented)) {
    job.push_back("-fstub-unimplemented");
  }
  if (const llvm::opt::Arg *arg = args.getLastArg(
          OPT_Wunselected_message, OPT_Wno_unselected_message)) {
    job.push_back(arg->getAsString(args));
  }
  for (const llvm::opt::Arg *arg : args.filtered(OPT_trace_flag_EQ)) {
    job.push_back(arg->getAsString(args));
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
// drivers give it: gfx1030's is amdgpu10.30-amd-amdhsa. The GPUs are for
// OpenMP's target regions, with -fopenmp, and for the process kinds that run
// on a GPU, with -fprocess-declarations.
static bool getOffloadArchs(
    const InputArgList &args, std::vector<OffloadArch> &archs) {
  std::vector<std::string> values{args.getAllArgValues(OPT_offload_arch_EQ)};
  if (!values.empty() && !args.hasArg(OPT_fopenmp) &&
      !args.hasArg(OPT_fprocess_declarations)) {
    reportError("--offload-arch needs -fopenmp or -fprocess-declarations");
    return false;
  }
  for (const std::string &value : values) {
    // A target ID, as Clang's: the processor, then features such as
    // gfx90a:xnack-.
    auto [processor, features]{llvm::StringRef{value}.split(':')};
    llvm::AMDGPU::GPUKind kind{llvm::AMDGPU::parseArchAMDGCN(processor)};
    if (kind == llvm::AMDGPU::GK_NONE) {
      reportError("offloading to '" + value +
          "' is not supported yet: only AMD GPUs, such as gfx1030, are");
      return false;
    }
    // Temporary (docs/INPROGRESS.md, A demo: a stencil whose active region
    // moves): features are reported until the device's bitcode carries them
    // as Clang's does, on each function, since the image's link gives LTO
    // only the processor. Code for the processor alone runs with xnack and
    // sramecc either on or off.
    if (!features.empty()) {
      reportError("--offload-arch=" + value +
          ": target features are not supported yet; give the processor "
          "alone, " +
          processor.str() + ", whose code runs with them on or off");
      return false;
    }
    llvm::Triple triple{kOffloadTargetAMDGPU};
    triple.setArch(triple.getArch(), llvm::AMDGPU::getSubArch(kind));
    archs.push_back({value, triple.str()});
  }
  return true;
}

namespace {
// The intermediate files of the jobs. Each is a unique temporary file, which
// the driver removes at the end, or with -save-temps, `<stem>.<suffix>` in
// the directory it names, kept.
class TemporaryFiles {
public:
  // The directory of -save-temps, or false after reporting an error.
  bool Configure(const InputArgList &args) {
    const llvm::opt::Arg *arg{args.getLastArg(OPT_save_temps_EQ)};
    if (!arg) {
      return true;
    }
    llvm::StringRef where{arg->getValue()};
    if (where != "cwd" && where != "obj") {
      reportError("-save-temps=" + where.str() +
          ": expected cwd or obj");
      return false;
    }
    keep_ = true;
    if (where == "obj") {
      dir_ = llvm::sys::path::parent_path(args.getLastArgValue(OPT_o)).str();
    }
    return true;
  }

  // A file's name, from `stem` and `suffix`.
  std::string Name(llvm::StringRef stem, llvm::StringRef suffix) {
    llvm::SmallString<128> name;
    if (keep_) {
      name = dir_;
      llvm::sys::path::append(name, stem + "." + suffix);
      return name.str().str();
    }
    llvm::sys::fs::getPotentiallyUniqueTempFileName(stem, suffix, name);
    removed_.push_back(name.str().str());
    return removed_.back();
  }

  void Remove() {
    for (const std::string &name : removed_) {
      llvm::sys::fs::remove(name);
    }
  }

private:
  bool keep_{false};
  std::string dir_;
  std::vector<std::string> removed_;
};
} // namespace

// The program that links: Clang, beside merkc or else on the PATH (design
// document, "Linking and offloading").
static std::string findLinker(llvm::StringRef executable) {
  std::string clang{findClang(executable)};
  return clang.empty() ? "clang" : clang;
}

// The directory of the compiled standard library and the runtime library:
// lib beside bin/merkc, in the build directory and the installation alike.
static std::string libraryDirectory(llvm::StringRef executable) {
  llvm::SmallString<128> dir{
      llvm::sys::path::parent_path(llvm::sys::path::parent_path(executable))};
  llvm::sys::path::append(dir, "lib");
  return dir.str().str();
}

// The jobs that make the image of `input`'s process kinds that run on a GPU,
// for each of `archs` (docs/INPROGRESS.md, GPU processes without omp target,
// step 4b): the kernels' bitcode, compiled without OpenMP; and the image, a
// shared object that Clang links from it with LTO, with the start object
// that exports what the offload plugin and the runtime look up, the
// device's runtime library, and libc and libm for the GPU, which Clang finds
// beside itself. The link optimizes at the -O given, or else at -O2, as the
// linker wrapper does for OpenMP's device code: AMDGPU's code generation is
// not well tested at -O0. At -O0 and -O1 it fails on a musttail call in
// divergent control flow ("failed to annotate CFG"), which code generation
// makes of a tail call between procedures of the same type (design document,
// LLVM bugs to report), but not in a process's code, whose branches are
// uniform (Lowering.cpp, MarkProcessCodeSingleLane). The host's compile of
// `input` embeds each image with the options that come back. A module without
// such kinds gets an image without kernels, which its host compile leaves out.
static std::vector<std::string> addProcessImageJobs(const InputArgList &args,
    llvm::StringRef executable, const std::string &input,
    const std::vector<OffloadArch> &archs, std::vector<Job> &jobs,
    TemporaryFiles &temporaries) {
  llvm::StringRef stem{llvm::sys::path::stem(input)};
  std::string libraryDir{libraryDirectory(executable)};
  std::vector<std::string> embeds;
  for (const OffloadArch &arch : archs) {
    // Bitcode named as an object, which Clang gives to the linker as it is,
    // for LTO, rather than compile it again.
    std::string kernels{
        temporaries.Name((stem + "-" + arch.arch + "-processes").str(), "o")};
    Job device{executable.str(), "-fc1", "-emit-llvm-bc"};
    addFrontendOptions(args, device, arch.triple, /*openMP=*/false);
    device.push_back("-fprocess-kernels");
    // The loops that the kinds run across teams keep their directives.
    if (args.hasArg(OPT_fopenmp)) {
      device.push_back("-fprocess-loops");
    }
    device.insert(device.end(), {"-o", kernels, input});
    jobs.push_back(std::move(device));
    std::string image{
        temporaries.Name((stem + "-" + arch.arch + "-processes").str(), "elf")};
    const llvm::opt::Arg *level{args.getLastArg(OPT_O)};
    Job link{findLinker(executable), "--target=amdgcn-amd-amdhsa",
        "-mcpu=" + arch.arch, "-flto",
        level ? level->getAsString(args) : std::string{"-O2"}};
    link.insert(link.end(),
        {kernels, libraryDir + "/merk-process-start-amdgcn.o",
            libraryDir + "/libmerkdevice-amdgcn.a", "-lc", "-lm", "-o", image});
    jobs.push_back(std::move(link));
    embeds.push_back("-fembed-process-image=" + arch.arch + ":" + image);
  }
  return embeds;
}

// The jobs that compile `input` to the object file `object` with its target
// regions offloaded to `archs`, as Flang's are: the host's bitcode, with its
// offload entries and the options in `hostOptions`; each GPU's, which
// matches them; the GPUs' code in one offload binary; and the host's object
// file, from its bitcode, with the offload binary embedded, for the linker
// wrapper to extract.
static void addOffloadJobs(const InputArgList &args, llvm::StringRef executable,
    const std::string &input, const std::string &object,
    const std::vector<OffloadArch> &archs,
    const std::vector<std::string> &hostOptions, std::vector<Job> &jobs,
    TemporaryFiles &temporaries) {
  llvm::StringRef stem{llvm::sys::path::stem(input)};
  std::string offloadTargets{"--offload-targets=" +
      std::string{kOffloadTargetAMDGPU}};
  std::string hostBitcode{temporaries.Name(stem, "bc")};
  Job host{executable.str(), "-fc1", "-emit-llvm-bc"};
  addFrontendOptions(args, host);
  host.insert(host.end(), hostOptions.begin(), hostOptions.end());
  host.insert(host.end(), {offloadTargets, "-o", hostBitcode, input});
  jobs.push_back(std::move(host));
  Job binary{findTool(executable, "llvm-offload-binary")};
  if (binary[0].empty()) {
    binary[0] = "llvm-offload-binary";
  }
  std::string offloadBinary{temporaries.Name(stem, "out")};
  binary.insert(binary.end(), {"-o", offloadBinary});
  for (const OffloadArch &arch : archs) {
    std::string deviceBitcode{
        temporaries.Name((stem + "-" + arch.arch).str(), "bc")};
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

// The jobs for the command line, or false after reporting an error: with
// -E, -S or -c, one -fc1 job per source; otherwise one per source, writing
// a temporary object file, and a link of the objects with the runtime
// library. With --offload-arch= and -fopenmp, each source's object file comes
// from the offload jobs instead; with --offload-arch= and
// -fprocess-declarations, the images of its GPU process kinds come first.
// `temporaries` names the intermediate files.
static bool buildJobs(const InputArgList &args, llvm::StringRef executable,
    std::vector<Job> &jobs, TemporaryFiles &temporaries) {
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
  bool offload{!archs.empty() && !preprocess && args.hasArg(OPT_fopenmp)};
  if (offload && assemble) {
    reportError("-S with --offload-arch is not supported yet");
    return false;
  }
  bool processImages{
      !archs.empty() && !preprocess && args.hasArg(OPT_fprocess_declarations)};
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
    std::vector<std::string> hostOptions;
    if (processImages) {
      hostOptions = addProcessImageJobs(
          args, executable, input, archs, jobs, temporaries);
    }
    if (offload) {
      std::string object{link ? temporaries.Name(
                                    llvm::sys::path::stem(input), "o")
              : !output.empty()
              ? output.str()
              : (llvm::sys::path::stem(input) + ".o").str()};
      if (link) {
        objects.push_back(object);
      }
      addOffloadJobs(args, executable, input, object, archs, hostOptions, jobs,
          temporaries);
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
    job.insert(job.end(), hostOptions.begin(), hostOptions.end());
    std::string jobOutput{link ? "" : output.str()};
    if (link) {
      jobOutput = temporaries.Name(llvm::sys::path::stem(input), "o");
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
    // The standard library's foreign code may call C's maths library.
    std::string libraryDir{libraryDirectory(executable)};
    Job job{findLinker(executable)};
    // -v shows the link's own jobs too, such as the linker wrapper's for
    // the GPUs' code, as Flang's -v does.
    if (args.hasArg(OPT_v)) {
      job.push_back("-v");
    }
    // Clang keeps the link's own intermediate files, such as the linker
    // wrapper's for the GPUs' code: the device link's LTO stages and its
    // image.
    if (const llvm::opt::Arg *arg = args.getLastArg(OPT_save_temps_EQ)) {
      job.push_back("-save-temps=" + std::string{arg->getValue()});
    }
    if (const llvm::opt::Arg *arg = args.getLastArg(OPT_target_EQ)) {
      job.push_back(arg->getAsString(args));
    }
    // OpenMP links as Clang does with -fopenmp, which adds the OpenMP
    // runtime library, and with --offload-arch=, which runs the linker
    // wrapper, which links the GPUs' code, and adds the offloading
    // libraries.
    if (args.hasArg(OPT_fopenmp)) {
      job.push_back("-fopenmp");
    }
    if (offload) {
      std::string offloadArch{"--offload-arch="};
      for (const OffloadArch &arch : archs) {
        offloadArch += (&arch == &archs.front() ? "" : ",") + arch.arch;
      }
      job.push_back(offloadArch);
      // The linker wrapper compiles the GPUs' code at the -O given, or else
      // at -O2, as Clang's and Flang's links do with the compile's -O.
      if (const llvm::opt::Arg *arg = args.getLastArg(OPT_O)) {
        job.push_back(arg->getAsString(args));
      }
      // The device's runtime, which target regions call to send and
      // receive, goes into the device image, as the linker wrapper links
      // the OpenMP device runtime. Only AMD GPUs are offloaded to so far.
      job.push_back("-Xoffload-linker-amdgcn-amd-amdhsa");
      job.push_back(libraryDir + "/libmerkdevice-amdgcn.a");
      // A kernel with statements around a parallel region stays generic:
      // OpenMPOpt's guarded SPMD form of one faults at address 0 when a value
      // read from memory before the region reaches it through a call
      // (test/CodeGen/omp-target-parallel-do.m).
      job.push_back("-Xoffload-linker-amdgcn-amd-amdhsa");
      job.push_back("-mllvm=-openmp-opt-disable-spmdization");
    }
    job.push_back("-o");
    job.push_back(output.empty() ? "a.out" : output.str());
    job.insert(job.end(), objects.begin(), objects.end());
    for (const std::string &dir : args.getAllArgValues(OPT_L)) {
      job.push_back("-L" + dir);
    }
    job.push_back("-L" + libraryDir);
    // The stack maps hold code addresses in a read-only section, which a
    // position-independent executable could only relocate at startup.
    job.push_back("-no-pie");
    job.push_back("-lmerkstd");
    job.push_back("-lmerkrt");
    // The runtime loads and launches GPU processes' kernels through LLVM's
    // offload library (runtime/gpu-processes.cpp), which -L must find when
    // it is not beside Clang.
    if (processImages) {
      job.push_back("-lLLVMOffload");
    }
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
  TemporaryFiles temporaries;
  if (!temporaries.Configure(*parsed) ||
      !buildJobs(*parsed, executable, jobs, temporaries)) {
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
  temporaries.Remove();
  return result;
}
