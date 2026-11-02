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
// this starts the runtime, calls it, and returns the program's exit
// status.
//
//===----------------------------------------------------------------------===//

extern "C" void merk_main(void);

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  merk_main();
  return 0;
}
