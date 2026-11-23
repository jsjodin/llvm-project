//===-- runtime/exception.cpp - Exceptions --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exceptions (docs/INPROGRESS.md, Exceptions). A try is a closure call of
// one of the entries below, which keeps a handler in its frame, on the
// process's list, and calls the goal; a throw jumps to the innermost
// handler, dropping the frames above it whole, as they hold nothing the
// collector or anything else needs. An uncaught exception is reported as
// mmc's programs report it, and ends the process, or the program in main.
// The library's own errors, such as that of a division by zero, are not
// thrown as values yet, and nothing catches them.
//
//===----------------------------------------------------------------------===//

#include "descriptor.h"
#include "process.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int omp_get_level(void) __attribute__((weak));

namespace merkurius::runtime {

// In the frame of a try's entry, at its stack pointer, where a throw puts the
// stack pointer back. The layout is the entries' (below).
struct TryFrame {
  TryFrame *previous;
  void *framePointer;
  void *resume;
  // The OpenMP level it was made at: a throw at another, inside a parallel
  // region that the goal started, cannot leave the region.
  std::int64_t level;
  std::uint32_t mxcsr;
  std::uint16_t control;
};

} // namespace merkurius::runtime

using namespace merkurius::runtime;

namespace {

// The tries of a thread that runs no process, as a parallel region's team
// threads.
thread_local TryFrame *threadTries{nullptr};

TryFrame *&Tries() {
  Process *process{CurrentProcess()};
  return process ? process->tries : threadTries;
}

std::int64_t Level() { return omp_get_level ? omp_get_level() : 0; }

} // namespace

extern "C" [[noreturn]] void merk_uncaught_exception(const char *exception);

extern "C" void merk_try_push(TryFrame *frame) {
  TryFrame *&tries{Tries()};
  frame->previous = tries;
  frame->level = Level();
  tries = frame;
}

extern "C" void merk_try_pop(TryFrame *frame) { Tries() = frame->previous; }

// The exception's value and its type_info are for the report if nothing
// catches it: mmc writes a software_error's text after "Software Error: ",
// and any other value as io.write does.
extern "C" [[noreturn]] void merk_throw(
    Word univ, Word typeInfo, Word value) {
  TryFrame *&tries{Tries()};
  TryFrame *frame{tries};
  if (!frame || frame->level != Level()) {
    // The text is also the reason a monitor of the process sees, failed(Text).
    char *text{nullptr};
    std::size_t size{0};
    std::FILE *stream{open_memstream(&text, &size)};
    if (!stream) {
      std::abort();
    }
    const auto *ctor{reinterpret_cast<const TypeCtor *>(typeInfo)};
    bool softwareError{(typeInfo & kTagMask) == 0 &&
        std::strcmp(ctor->module, "exception") == 0 &&
        std::strcmp(ctor->name, "software_error") == 0};
    if (softwareError) {
      std::fprintf(stream, "Software Error: %s",
          reinterpret_cast<const char *>(value));
    } else {
      WriteValue(stream, typeInfo, value);
    }
    std::fclose(stream);
    merk_uncaught_exception(text);
  }
  tries = frame->previous;
#if defined(__x86_64__)
  __asm__ volatile("movq %0, %%rsp\n\t"
                   "movq %1, %%rbp\n\t"
                   "jmpq *%2"
                   :
                   : "c"(frame), "d"(frame->framePointer), "S"(frame->resume),
                   "a"(univ)
                   : "memory");
  __builtin_unreachable();
#else
  // Temporary (docs/INPROGRESS.md, Exceptions, step 3): a try on another
  // target, as a GPU, which has no entries yet, makes no TryFrame.
  std::abort();
#endif
}

#if defined(__x86_64__)

// The entries of a try, which the library calls as closures: in the tail
// calling convention, with the closure, the goal and, for try_io, the io
// state's word, and giving back words in rax, rdx and rcx.
// - merk_try_det: a det or cc_multi pred(out); gives the value and the
//   exception, a univ, or 0 if none.
// - merk_try_semidet: a semidet or cc_nondet pred(out); gives 0, 1 if the
//   goal failed, or 2 if it threw, then the value, or 0, and the exception.
// - merk_try_io: a det or cc_multi pred(out, di, uo); gives the value, the
//   exception, or 0, and the io state's word.
// A callee in the tail calling convention pops 8 bytes above its return
// address, which its caller gives back after the call, as the stack stays
// aligned to 16 bytes: so an entry returns with ret $8, and calls the goal
// with an unused 8 bytes under the TryFrame, which the goal may write.
// Each keeps its frame pointer, as a Merkurius frame does, so that the
// collector's walk goes from the goal's frames through this one to the
// caller's (roots.cpp, kTryReturns), and saves every callee-saved register,
// which a throw gets back from here. The TryFrame is at the stack pointer,
// 80 bytes below the frame pointer, under the five registers.
#define MERK_TRY_ENTRY(name, resume)                                           \
  "    .p2align 4\n"                                                           \
  "    .globl " #name "\n"                                                     \
  "    .hidden " #name "\n"                                                    \
  "    .type " #name ", @function\n"                                           \
  #name ":\n"                                                                  \
  "    pushq %rbp\n"                                                           \
  "    movq %rsp, %rbp\n"                                                      \
  "    pushq %rbx\n"                                                           \
  "    pushq %r12\n"                                                           \
  "    pushq %r13\n"                                                           \
  "    pushq %r14\n"                                                           \
  "    pushq %r15\n"                                                           \
  "    subq $40, %rsp\n"                                                       \
  "    movq %rsi, %r12\n"                                                      \
  "    movq %rdx, %r13\n"                                                      \
  "    movq %rbp, 8(%rsp)\n"                                                   \
  "    leaq " #resume "(%rip), %rax\n"                                         \
  "    movq %rax, 16(%rsp)\n"                                                  \
  "    stmxcsr 32(%rsp)\n"                                                     \
  "    fnstcw 36(%rsp)\n"                                                      \
  "    movq %rsp, %rdi\n"                                                      \
  "    callq merk_try_push\n"                                                  \
  "    movq %r12, %rdi\n"                                                      \
  "    movq %r13, %rsi\n"                                                      \
  "    subq $16, %rsp\n"                                                       \
  "    callq *(%rdi)\n"                                                        \
  "    .globl " #name "_return\n"                                              \
  "    .hidden " #name "_return\n"                                             \
  #name "_return:\n"                                                           \
  "    leaq -80(%rbp), %rsp\n"

#define MERK_TRY_RESUME(resume)                                                \
  #resume ":\n"                                                                \
  "    ldmxcsr 32(%rsp)\n"                                                     \
  "    fldcw 36(%rsp)\n"

#define MERK_TRY_EPILOGUE(name)                                                \
  "1:\n"                                                                       \
  "    leaq -40(%rbp), %rsp\n"                                                 \
  "    popq %r15\n"                                                            \
  "    popq %r14\n"                                                            \
  "    popq %r13\n"                                                            \
  "    popq %r12\n"                                                            \
  "    popq %rbx\n"                                                            \
  "    popq %rbp\n"                                                            \
  "    ret $8\n"                                                               \
  "    .size " #name ", .-" #name "\n"

__asm__("    .text\n"
        // The value in rax.
        MERK_TRY_ENTRY(merk_try_det, .Lmerk_try_det_resume)
        "    movq %rax, %r12\n"
        "    movq %rsp, %rdi\n"
        "    callq merk_try_pop\n"
        "    movq %r12, %rax\n"
        "    xorl %edx, %edx\n"
        "    jmp 1f\n"
        MERK_TRY_RESUME(.Lmerk_try_det_resume)
        "    movq %rax, %rdx\n"
        "    xorl %eax, %eax\n"
        MERK_TRY_EPILOGUE(merk_try_det)

        // Whether it succeeded in al, the value in rdx.
        MERK_TRY_ENTRY(merk_try_semidet, .Lmerk_try_semidet_resume)
        "    movq %rdx, %r12\n"
        "    movl %eax, %r13d\n"
        "    movq %rsp, %rdi\n"
        "    callq merk_try_pop\n"
        "    xorl %ecx, %ecx\n"
        "    testb $1, %r13b\n"
        "    jz 2f\n"
        "    xorl %eax, %eax\n"
        "    movq %r12, %rdx\n"
        "    jmp 1f\n"
        "2:\n"
        "    movl $1, %eax\n"
        "    xorl %edx, %edx\n"
        "    jmp 1f\n"
        MERK_TRY_RESUME(.Lmerk_try_semidet_resume)
        "    movq %rax, %rcx\n"
        "    movl $2, %eax\n"
        "    xorl %edx, %edx\n"
        MERK_TRY_EPILOGUE(merk_try_semidet)

        // The value in rax, the io state's word in rdx.
        MERK_TRY_ENTRY(merk_try_io, .Lmerk_try_io_resume)
        "    movq %rax, %r12\n"
        "    movq %rdx, %r13\n"
        "    movq %rsp, %rdi\n"
        "    callq merk_try_pop\n"
        "    movq %r12, %rax\n"
        "    xorl %edx, %edx\n"
        "    movq %r13, %rcx\n"
        "    jmp 1f\n"
        MERK_TRY_RESUME(.Lmerk_try_io_resume)
        "    movq %rax, %rdx\n"
        "    xorl %eax, %eax\n"
        "    xorl %ecx, %ecx\n"
        MERK_TRY_EPILOGUE(merk_try_io));

extern "C" void merk_try_det(void);
extern "C" void merk_try_semidet(void);
extern "C" void merk_try_io(void);

// The entries as closures of nothing captured: a closure's cell has its
// entry's code, then the layout of what it captured, which a cell outside
// the heap does not need, as the collector neither moves nor reads it.
extern "C" const Word merk_try_det_closure[2]{
    reinterpret_cast<Word>(&merk_try_det), 0};
extern "C" const Word merk_try_semidet_closure[2]{
    reinterpret_cast<Word>(&merk_try_semidet), 0};
extern "C" const Word merk_try_io_closure[2]{
    reinterpret_cast<Word>(&merk_try_io), 0};

#endif

extern "C" [[noreturn]] void merk_uncaught_exception(const char *exception) {
  std::fflush(stdout);
  std::fprintf(stderr, "Uncaught Merkurius exception:\n%s\n", exception);
  merkurius::runtime::EndAfterFailure(exception);
  merkurius::runtime::ExitProgram(1);
}

// Temporary (chunk 1 step 6, docs/INPROGRESS.md): the body of a library
// procedure that code generation cannot make yet, with
// -fstub-unimplemented, such as a nondet mode; it goes when code generation
// makes every procedure of the library.
extern "C" [[noreturn]] void merk_not_implemented(const char *what) {
  std::fflush(stdout);
  std::fprintf(stderr, "Not yet implemented: %s\n", what);
  merkurius::runtime::ExitProgram(1);
}
