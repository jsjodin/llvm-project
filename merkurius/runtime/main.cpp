//===-- runtime/main.cpp - The program's entry ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The C main of every Mercury program (design document, "Runtime system").
// The compiler gives the main module's main(di, uo) a C entry, merk_main;
// this starts the runtime, saving the environment that the run time
// conditions of trace goals test (trace.cpp) and the command line, which
// io.command_line_arguments and io.progname read, and the host's side of
// target regions (offload.cpp), calls it, and returns the program's exit
// status, 0 unless io.set_exit_status set another.
//
//===----------------------------------------------------------------------===//

#include "process.h"

#include <cstdint>
#include <cstdlib>

extern "C" void merk_main(void);
extern "C" void merk_save_environment(void);
extern "C" void merk_offload_start(void);
extern "C" void merk_process_run_main(void (*entry)(void));
extern "C" void merk_process_stop(void);

namespace {
int argumentCount{0};
char **arguments{nullptr};
int exitStatus{0};
} // namespace

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
  merk_main();
  merk_process_stop();
  merkurius::runtime::CheckPinnedAtEnd();
  merkurius::runtime::ExitProgram(exitStatus);
}

int main(int argc, char **argv) {
  argumentCount = argc;
  arguments = argv;
  merk_save_environment();
  merk_offload_start();
  merk_process_run_main(EndWithMain);
  return exitStatus;
}
