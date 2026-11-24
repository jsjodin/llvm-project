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

// The descriptors of the library's own errors, software_error, domain_error
// and index_out_of_bounds, which exception.m's foreign code lists. Weak: a
// program that never calls try does not link exception.m, and nothing
// catches its errors.
extern "C" const Word merk_library_error_types[] __attribute__((weak));

namespace {

// A quoted string as mmc writes it, from after its opening quote to before
// its closing one, into a string of its own. False if it is not one.
bool Unquote(const char *from, const char *end, char *&text) {
  text = static_cast<char *>(std::malloc(end - from + 1));
  char *to{text};
  while (from < end) {
    char c{*from++};
    if (c == '\\') {
      if (from == end) {
        return false;
      }
      c = *from++;
      c = c == 'n' ? '\n' : c == 't' ? '\t' : c;
    }
    *to++ = c;
  }
  *to = '\0';
  return true;
}

} // namespace

// An error of the library's own, with its text as mmc reports it uncaught:
// "Software Error: " and the message, domain_error("message"), or
// index_out_of_bounds("message"). It is thrown as that value, so that a try
// catches it; if it is none of them, or the program links no try, it is
// reported. Code generation's checks call it (merk.throw_if), and the
// library's C, from Merkurius code, which holds no lock a jump would leave
// held, as a call from the runtime's own C++ may. The univ's cell and the
// message are outside the heap, which the collector neither moves nor
// frees: an error's few bytes are not given back.
extern "C" [[noreturn]] void merk_library_error(const char *text) {
  static constexpr char kSoftware[]{"Software Error: "};
  static constexpr char kDomain[]{"domain_error(\""};
  static constexpr char kIndex[]{"index_out_of_bounds(\""};
  if (!merk_library_error_types) {
    merk_uncaught_exception(text);
  }
  std::size_t length{std::strlen(text)};
  Word type{0};
  char *message{nullptr};
  if (std::strncmp(text, kSoftware, sizeof kSoftware - 1) == 0) {
    type = merk_library_error_types[0];
    message = strdup(text + sizeof kSoftware - 1);
  } else {
    for (int kind{1}; kind <= 2; ++kind) {
      const char *prefix{kind == 1 ? kDomain : kIndex};
      std::size_t size{std::strlen(prefix)};
      if (length >= size + 2 && std::strncmp(text, prefix, size) == 0 &&
          std::strcmp(text + length - 2, "\")") == 0 &&
          Unquote(text + size, text + length - 2, message)) {
        type = merk_library_error_types[kind];
        break;
      }
    }
  }
  if (!type) {
    merk_uncaught_exception(text);
  }
  auto *univ{static_cast<Word *>(std::malloc(2 * sizeof(Word)))};
  univ[0] = type;
  univ[1] = reinterpret_cast<Word>(message);
  merk_throw(reinterpret_cast<Word>(univ), type, univ[1]);
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
        MERK_TRY_EPILOGUE(merk_try_io)

        // merk_try_nondet: a nondet or multi pred(out), for try_all, called
        // as a nondet closure, with the closure, the goal, the continuation
        // and its environment. It calls the goal with merk_try_nondet_solution
        // as the continuation and, as its environment, the continuation and
        // the environment it was given, stored above the TryFrame. Each
        // solution goes on to the continuation as (0, the value, 0), with the
        // TryFrame still on the list; if the goal throws, the exception goes
        // as (1, 0, the univ), in a tail call, and the goal gives no more.
        // The continuation's flag, whether to stop, comes back in al. An
        // exception that the continuation throws is caught here too, so its
        // caller's continuation must not throw, as try_all's does not: it
        // only adds the solution to the list (exception.m).
        "    .p2align 4\n"
        "    .globl merk_try_nondet\n"
        "    .hidden merk_try_nondet\n"
        "    .type merk_try_nondet, @function\n"
        "merk_try_nondet:\n"
        "    pushq %rbp\n"
        "    movq %rsp, %rbp\n"
        "    pushq %rbx\n"
        "    pushq %r12\n"
        "    pushq %r13\n"
        "    pushq %r14\n"
        "    pushq %r15\n"
        "    subq $56, %rsp\n"
        "    movq %rsi, %r12\n"
        "    movq %rdx, 40(%rsp)\n"
        "    movq %rcx, 48(%rsp)\n"
        "    movq %rbp, 8(%rsp)\n"
        "    leaq .Lmerk_try_nondet_resume(%rip), %rax\n"
        "    movq %rax, 16(%rsp)\n"
        "    stmxcsr 32(%rsp)\n"
        "    fnstcw 36(%rsp)\n"
        "    movq %rsp, %rdi\n"
        "    callq merk_try_push\n"
        "    movq %r12, %rdi\n"
        "    leaq merk_try_nondet_solution(%rip), %rsi\n"
        "    leaq 40(%rsp), %rdx\n"
        "    subq $16, %rsp\n"
        "    callq *(%rdi)\n"
        "    .globl merk_try_nondet_return\n"
        "    .hidden merk_try_nondet_return\n"
        "merk_try_nondet_return:\n"
        "    leaq -96(%rbp), %rsp\n"
        "    movl %eax, %r13d\n"
        "    movq %rsp, %rdi\n"
        "    callq merk_try_pop\n"
        "    movl %r13d, %eax\n"
        "    leaq -40(%rbp), %rsp\n"
        "    popq %r15\n"
        "    popq %r14\n"
        "    popq %r13\n"
        "    popq %r12\n"
        "    popq %rbx\n"
        "    popq %rbp\n"
        "    ret $8\n"
        ".Lmerk_try_nondet_resume:\n"
        "    ldmxcsr 32(%rsp)\n"
        "    fldcw 36(%rsp)\n"
        "    movq 40(%rsp), %r10\n"
        "    movq 48(%rsp), %rdi\n"
        "    movl $1, %esi\n"
        "    xorl %edx, %edx\n"
        "    movq %rax, %rcx\n"
        "    leaq -40(%rbp), %rsp\n"
        "    popq %r15\n"
        "    popq %r14\n"
        "    popq %r13\n"
        "    popq %r12\n"
        "    popq %rbx\n"
        "    popq %rbp\n"
        "    jmpq *%r10\n"
        "    .size merk_try_nondet, .-merk_try_nondet\n"

        // A solution of merk_try_nondet's goal, which goes on to the
        // continuation in its environment, as a tail call.
        "    .p2align 4\n"
        "    .type merk_try_nondet_solution, @function\n"
        "merk_try_nondet_solution:\n"
        "    movq %rsi, %rdx\n"
        "    movq (%rdi), %r10\n"
        "    movq 8(%rdi), %rdi\n"
        "    xorl %esi, %esi\n"
        "    xorl %ecx, %ecx\n"
        "    jmpq *%r10\n"
        "    .size merk_try_nondet_solution, .-merk_try_nondet_solution\n");

extern "C" void merk_try_det(void);
extern "C" void merk_try_semidet(void);
extern "C" void merk_try_io(void);
extern "C" void merk_try_nondet(void);

// The entries as closures of nothing captured: a closure's cell has its
// entry's code, then the layout of what it captured, which a cell outside
// the heap does not need, as the collector neither moves nor reads it.
extern "C" const Word merk_try_det_closure[2]{
    reinterpret_cast<Word>(&merk_try_det), 0};
extern "C" const Word merk_try_semidet_closure[2]{
    reinterpret_cast<Word>(&merk_try_semidet), 0};
extern "C" const Word merk_try_io_closure[2]{
    reinterpret_cast<Word>(&merk_try_io), 0};
extern "C" const Word merk_try_nondet_closure[2]{
    reinterpret_cast<Word>(&merk_try_nondet), 0};

#endif

// 1 if the program was compiled with -fmercury-messages, whose main module
// defines it: the messages that mmc's programs have too are printed as
// those print them.
extern "C" __attribute__((weak)) const std::int32_t merk_mercury_messages{0};

const char *merkurius::runtime::UncaughtExceptionHeader() {
  // Read through a volatile pointer, so that the weak definition's 0 is not
  // folded in.
  const volatile std::int32_t *flag{&merk_mercury_messages};
  return *flag ? "Uncaught Mercury exception:" : "Uncaught Merkurius exception:";
}

extern "C" [[noreturn]] void merk_uncaught_exception(const char *exception) {
  std::fflush(stdout);
  std::fprintf(stderr, "%s\n%s\n",
      merkurius::runtime::UncaughtExceptionHeader(), exception);
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
