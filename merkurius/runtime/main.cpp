//===-- runtime/main.cpp - The program's entry ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The C main of every Merkurius program (design document, "Runtime system").
// The compiler gives the main module's main(di, uo) a C entry, merk_main;
// this starts the runtime, calls the initialise directives' procedures
// before it and the finalise directives' after, saving the environment that the run time
// conditions of trace goals test (trace.cpp) and the command line, which
// io.command_line_arguments and io.progname read, and the host's side of
// target regions (offload.cpp), calls it, and returns the program's exit
// status, 0 unless io.set_exit_status set another.
//
//===----------------------------------------------------------------------===//

#include "process.h"

#include <cstdint>
#include <cstdlib>
#include <malloc.h>

extern "C" void merk_main(void);
extern "C" void merk_save_environment(void);
extern "C" void merk_offload_start(void);
extern "C" void merk_process_run_main(void (*entry)(void));
extern "C" void merk_process_stop(void);

namespace {
int argumentCount{0};
char **arguments{nullptr};
int exitStatus{0};
// The procedures of the initialise and finalise directives (reference
// manual, "Module initialisation" and "Module finalisation"), each module's
// in their order. Lists of zero-initialised statics, as modules'
// constructors register them before any of this file's would be
// constructed.
struct Initialiser {
  void (*procedure)(void);
  Initialiser *next;
};
struct Initialisers {
  Initialiser *first;
  Initialiser **last;
};
Initialisers initialisers, finalisers;

void RunAll(const Initialisers &list) {
  for (Initialiser *i{list.first}; i; i = i->next) {
    i->procedure();
  }
}
} // namespace

// Called by each module's C constructor, before main.
extern "C" void merk_register_initialiser(void (*procedure)(void),
    std::int32_t final) {
  Initialisers &list{final ? finalisers : initialisers};
  auto *added{static_cast<Initialiser *>(std::malloc(sizeof(Initialiser)))};
  if (!added) {
    std::abort();
  }
  *added = {procedure, nullptr};
  *(list.last ? list.last : &list.first) = added;
  list.last = &added->next;
}

// io.set_exit_status: the status main returns.
extern "C" void merk_set_exit_status(std::int64_t status) {
  exitStatus = static_cast<int>(status);
}

// The number of words on the command line, the program's name first.
extern "C" std::int64_t merk_argument_count(void) { return argumentCount; }

// The command line's word at `index`, which lives as long as the program, so
// the string is not copied.
extern "C" char *merk_argument(std::int64_t index) { return arguments[index]; }

// The program ends with main, on whichever thread it then runs.
static void EndWithMain(void) {
  RunAll(initialisers);
  merk_main();
  RunAll(finalisers);
  merk_process_stop();
  merkurius::runtime::CheckFreedAtEnd();
  merkurius::runtime::ExitProgram(exitStatus);
}

int main(int argc, char **argv) {
  // The heap's chunks come from malloc, and each collection frees a
  // from-space and takes a to-space. glibc would map each chunk of a
  // megabyte on its own, or give the top of its heap back to the system
  // when the chunks there are freed: either way the next chunks' pages are
  // faulted in again, which took more time than the collections' copying.
  // Chunks come from its heap, which keeps what is freed at its top.
  mallopt(M_MMAP_THRESHOLD, 32 << 20);
  mallopt(M_TRIM_THRESHOLD, 64 << 20);
  argumentCount = argc;
  arguments = argv;
  merk_save_environment();
  merk_offload_start();
  merk_process_run_main(EndWithMain);
  return exitStatus;
}
