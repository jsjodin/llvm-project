//===-- runtime/device/process-start.cpp - A GPU process image's start ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the image of a module's GPU process kinds must export, which the
// driver links into each such image as an object of its own, not a member
// of an archive (docs/INPROGRESS.md, GPU processes without omp target,
// step 4b). With LTO, a symbol is exported only if an object given to the
// link defines it: a definition from an archive, such as libc's client or
// libmerkdevice's device number, is internalized (step 1).
//
// - libc's RPC client, which the offload plugin finds by its dynamic symbol
//   and points at the host's buffer when it loads the image. OpenMP's device
//   runtime defines it the same way for a region's image.
// - The device's number, which the host's runtime sets when it loads the
//   image (device.cpp).
// - That this is a GPU process's image, so that the device's runtime takes
//   the process's arena from its kernel's arguments (device.cpp). It is not
//   exported: libmerkdevice's weak definition, 0, is left out.
// - omp_get_device_num, which device.cpp refers to weakly for a region's
//   image, where OpenMP's device runtime defines it. An undefined dynamic
//   symbol, even a weak one, stops HSA's loader from loading the image, so
//   the start object defines it, hidden. It is called only while the
//   device's number is -1, which it is not once the runtime has loaded the
//   image.
//
//===----------------------------------------------------------------------===//

#include "shared/rpc.h"

extern "C"
    [[gnu::visibility("protected"), gnu::weak]] rpc::Client __llvm_rpc_client;

extern "C"
    [[gnu::visibility("protected"), gnu::weak]] int merk_device_number{-1};

extern "C" [[gnu::visibility("hidden")]] int merk_device_process_image{1};

extern "C" [[gnu::visibility("hidden")]] int omp_get_device_num(void) {
  return -1;
}
