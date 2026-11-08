//===-- runtime/stack.h - Processes' stacks and switching -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A process's stack and the switch between stacks (design document, "Processes
// and scheduler"; docs/INPROGRESS.md, The scheduler, step 2). A stack has a
// fixed size and never moves. It is a slot of a large mapping with
// MAP_NORESERVE, whose pages the kernel commits as they are touched, so a
// slot's size costs only address space. Sizes are rounded up to a power of
// two, each with a free list, so that reusing a slot needs no system call.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_STACK_H_
#define MERKURIUS_RUNTIME_STACK_H_

#include <cstddef>
#include <cstdint>

namespace merkurius::runtime {

constexpr std::size_t kDefaultStackBytes{std::size_t{64} << 20};
// How far above the base of a stack the overflow check stops, for frames
// that are not checked and foreign code.
constexpr std::size_t kStackMargin{32 * 1024};

struct Stack {
  char *base; // its lowest address
  std::size_t bytes;
  // What the overflow check compares the stack pointer with: a margin above
  // the base, for frames that are not checked and foreign code.
  char *limit;
  Stack *nextFree;
};

// A stack of at least `bytes`, or null if the address space is exhausted.
Stack *NewStack(std::size_t bytes);
// Its slot goes back to the free list for its size. Its pages stay committed.
void FreeStack(Stack *);

// Where a suspended computation goes on from.
struct Context {
  void *stackPointer;
};

// A context that starts `body(argument)` on the stack when it is switched
// to. The body must not return: it ends by switching away for good.
Context NewContext(Stack &, void (*body)(void *), void *argument);

// Suspends the caller in `from`, and goes on from `to`.
void SwitchContext(Context &from, Context &to);

// Calls a closure of pred(T, io, io) with the word `argument`. Closures'
// entries are in the tail calling convention, whose callee may pop from the
// stack, which C++ cannot call.
void CallClosure(std::uint64_t closure, std::uint64_t argument);

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_STACK_H_
