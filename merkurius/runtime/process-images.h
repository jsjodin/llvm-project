//===-- runtime/process-images.h - GPU process kinds' images ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The images of the modules' process kinds that run on a GPU, one for each
// module and GPU, which each module's start-up code registers
// (docs/INPROGRESS.md, GPU processes without omp target, step 4b). An image
// is a shared object for the GPU, with a kernel merk_process_<kind> for
// each such kind of the module. The runtime loads it when a process is
// first spawned on a GPU of its kind (step 4c).
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_PROCESS_IMAGES_H_
#define MERKURIUS_RUNTIME_PROCESS_IMAGES_H_

#include <cstdint>

namespace merkurius::runtime {

struct ProcessImage {
  const char *module; // the module's name, as Mercury writes it
  const char *gpu; // the GPU it is for, as --offload-arch names it: gfx1030
  const void *image;
  std::uint64_t bytes;
};

// The image of `module`'s kinds for `gpu`, or null if it registered none.
const ProcessImage *FindProcessImage(const char *module, const char *gpu);

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_PROCESS_IMAGES_H_
