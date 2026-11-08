//===-- runtime/stack.cpp - Processes' stacks and switching ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "stack.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <sys/mman.h>

namespace merkurius::runtime {
namespace {

constexpr unsigned kSmallestShift{16}; // 64 KiB
constexpr unsigned kLargestShift{40};
constexpr std::size_t kFirstSlabSlots{64};
constexpr std::size_t kLargestSlabSlots{4096};

// One size's slots: those freed, and what is left of its latest mapping.
struct SizeClass {
  Stack *free;
  char *next;
  std::size_t left;
  std::size_t slabSlots;
};

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
SizeClass classes[kLargestShift + 1];

unsigned ShiftFor(std::size_t bytes) {
  unsigned shift{kSmallestShift};
  while (shift <= kLargestShift && (std::size_t{1} << shift) < bytes) {
    ++shift;
  }
  return shift;
}

// A new mapping for the size. A limit on the address space, such as ulimit
// -v, can refuse a large one, so it asks for less until one fits.
bool Grow(SizeClass &sizeClass, std::size_t bytes) {
  std::size_t slots{sizeClass.slabSlots ? sizeClass.slabSlots : kFirstSlabSlots};
  for (; slots != 0; slots /= 2) {
    void *mapping{mmap(nullptr, slots * bytes, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0)};
    if (mapping != MAP_FAILED) {
      sizeClass.next = static_cast<char *>(mapping);
      sizeClass.left = slots;
      sizeClass.slabSlots =
          slots * 2 < kLargestSlabSlots ? slots * 2 : kLargestSlabSlots;
      return true;
    }
  }
  return false;
}

} // namespace

Stack *NewStack(std::size_t bytes) {
  unsigned shift{ShiftFor(bytes)};
  if (shift > kLargestShift) {
    return nullptr;
  }
  std::size_t size{std::size_t{1} << shift};
  SizeClass &sizeClass{classes[shift]};
  pthread_mutex_lock(&lock);
  Stack *stack{sizeClass.free};
  if (stack) {
    sizeClass.free = stack->nextFree;
  } else if (sizeClass.left != 0 || Grow(sizeClass, size)) {
    stack = static_cast<Stack *>(std::malloc(sizeof(Stack)));
    if (stack) {
      *stack = {sizeClass.next, size, sizeClass.next + kStackMargin, nullptr};
      sizeClass.next += size;
      --sizeClass.left;
    }
  }
  pthread_mutex_unlock(&lock);
  return stack;
}

void FreeStack(Stack *stack) {
  SizeClass &sizeClass{classes[ShiftFor(stack->bytes)]};
  pthread_mutex_lock(&lock);
  stack->nextFree = sizeClass.free;
  sizeClass.free = stack;
  pthread_mutex_unlock(&lock);
}

} // namespace merkurius::runtime

using merkurius::runtime::Context;

#if defined(__x86_64__)

// The callee-saved registers of the System V ABI, with the control words of
// the floating-point units, which it also has the callee keep.
extern "C" void merk_switch_stack(void **save, void *to);
extern "C" void merk_stack_entry(void);

__asm__(R"(
    .text
    .p2align 4
    .globl merk_switch_stack
    .hidden merk_switch_stack
    .type merk_switch_stack, @function
merk_switch_stack:
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    subq $8, %rsp
    stmxcsr (%rsp)
    fnstcw 4(%rsp)
    movq %rsp, (%rdi)
    movq %rsi, %rsp
    ldmxcsr (%rsp)
    fldcw 4(%rsp)
    addq $8, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret
    .size merk_switch_stack, .-merk_switch_stack

    .p2align 4
    .globl merk_stack_entry
    .hidden merk_stack_entry
    .type merk_stack_entry, @function
merk_stack_entry:
    .cfi_startproc
    .cfi_undefined rip
    movq %r13, %rdi
    callq *%r12
    ud2
    .cfi_endproc
    .size merk_stack_entry, .-merk_stack_entry
)");

namespace merkurius::runtime {

Context NewContext(Stack &stack, void (*body)(void *), void *argument) {
  // After the switch's return the stack pointer is `top`, aligned as the
  // call in merk_stack_entry needs it.
  auto top{reinterpret_cast<std::uintptr_t>(stack.base + stack.bytes - 16) &
      ~std::uintptr_t{15}};
  auto *frame{reinterpret_cast<std::uint64_t *>(top) - 8};
  std::uint32_t mxcsr;
  std::uint16_t control;
  __asm__("stmxcsr %0\n\tfnstcw %1" : "=m"(mxcsr), "=m"(control));
  frame[0] = mxcsr | std::uint64_t{control} << 32;
  frame[1] = 0; // r15
  frame[2] = 0; // r14
  frame[3] = reinterpret_cast<std::uint64_t>(argument); // r13
  frame[4] = reinterpret_cast<std::uint64_t>(body); // r12
  frame[5] = 0; // rbx
  frame[6] = 0; // rbp
  frame[7] = reinterpret_cast<std::uint64_t>(&merk_stack_entry);
  return {frame};
}

void SwitchContext(Context &from, Context &to) {
  merk_switch_stack(&from.stackPointer, to.stackPointer);
}

} // namespace merkurius::runtime

#else

namespace merkurius::runtime {

[[noreturn]] static void NoSwitch() {
  std::fflush(stdout);
  std::fprintf(stderr,
      "Mercury runtime: switching stacks is not implemented on this "
      "architecture\n");
  std::exit(1);
}

Context NewContext(Stack &, void (*)(void *), void *) { NoSwitch(); }
void SwitchContext(Context &, Context &) { NoSwitch(); }

} // namespace merkurius::runtime

#endif
