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

// Whether MERK_GC_EVERY_ALLOC asks for a collection at each allocation, or
// at each N-th if it is a number N, for testing.
bool CollectsEveryAllocation();

// From merk_alloc_slow: collects the running process's heap if it may be
// collected and is due, and the allocation is from Mercury code at a safepoint,
// and whether it did. A C++ caller holds raw pointers, so the heap just grows.
bool CollectIfDue(void *frame);

// Walks the stack from the frame of merk_alloc_slow, checks each root against
// its type, and aborts on one that does not fit. No collection happens.
void CheckRoots(void *frame);

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_ROOTS_H_
