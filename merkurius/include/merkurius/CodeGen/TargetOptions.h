//===-- include/merkurius/CodeGen/TargetOptions.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_CODEGEN_TARGETOPTIONS_H_
#define MERKURIUS_CODEGEN_TARGETOPTIONS_H_

// The options that choose the target (TargetInfo) and the optimization
// level.

#include <string>
#include <vector>

namespace merkurius::codegen {

struct TargetOptions {
  std::string triple; // -triple; empty for the host's
  std::string cpu; // -target-cpu
  std::string features; // -target-feature, joined with commas
  unsigned optimizationLevel{0}; // -O
  // AMDGPU's code object version, as Clang's -mcode-object-version= gives
  // it: by default 6, which ROCm's device libraries (oclc_abi_version_600)
  // and Clang's default agree on.
  unsigned codeObjectVersion{600};
  // OpenMP offloading (design document, "Linking and offloading"), as in
  // Clang's and Flang's -fc1: -fopenmp and -fopenmp-version=, which code
  // generation needs as the front end does; -fopenmp-is-target-device, for
  // the device compile, with the host's LLVM IR from
  // -fopenmp-host-ir-file-path; and -fopenmp-targets=, the device triples
  // the host compile makes offload entries for.
  bool openMP{false};
  unsigned openMPVersion{52};
  bool openMPIsTargetDevice{false};
  std::string openMPHostIRFile;
  std::vector<std::string> openMPTargets;
  // -fembed-offload-object=: offload binaries to embed in the host object.
  std::vector<std::string> offloadObjects;
  // GPU processes (docs/INPROGRESS.md, GPU processes without omp target):
  // -fprocess-kernels, for the device compile that makes a kernel for each
  // process kind that runs on a GPU, without OpenMP; and
  // -fembed-process-image=, the images of those kernels, one for each GPU,
  // which the host's compile embeds and registers.
  bool processKernels{false};
  std::vector<std::string> processImages;

  // Whether the compile is for a device: OpenMP's device compile or GPU
  // processes'.
  bool IsDevice() const { return openMPIsTargetDevice || processKernels; }
};

} // namespace merkurius::codegen

#endif // MERKURIUS_CODEGEN_TARGETOPTIONS_H_
