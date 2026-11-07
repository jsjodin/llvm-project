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
// conditions of trace goals test (trace.cpp), calls it, and returns the
// program's exit status.
//
//===----------------------------------------------------------------------===//

extern "C" void merk_main(void);
extern "C" void merk_save_environment(void);

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  merk_save_environment();
  merk_main();
  return 0;
}
