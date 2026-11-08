//===-- runtime/roots.h - Finding a process's roots -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The stack maps and each module's table of root types, and a walk of a
// stack's safepoints (docs/INPROGRESS.md, Statepoints and stack maps).
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_ROOTS_H_
#define MERKURIUS_RUNTIME_ROOTS_H_

namespace merkurius::runtime {

// Whether MERK_CHECK_ROOTS asks each safepoint to be checked.
bool CheckRootsEnabled();

// Walks the stack from the frame of merk_alloc_slow, checks each root against
// its type, and aborts on one that does not fit. No collection happens.
void CheckRoots(void *frame);

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_ROOTS_H_
