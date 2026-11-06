//===-- runtime/exception.cpp - Exceptions --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exceptions, so far only those the library throws, such as that of a
// division by zero, which nothing catches yet: the program reports the
// exception as mmc's programs do, and exits with status 1.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>

extern "C" [[noreturn]] void merk_uncaught_exception(const char *exception) {
  std::fflush(stdout);
  std::fprintf(stderr, "Uncaught Mercury exception:\n%s\n", exception);
  std::exit(1);
}

// Temporary (chunk 1 step 6, docs/INPROGRESS.md): the body of a library
// procedure that code generation cannot make yet, with
// -fstub-unimplemented, such as a nondet mode; it goes when code generation
// makes every procedure of the library.
extern "C" [[noreturn]] void merk_not_implemented(const char *what) {
  std::fflush(stdout);
  std::fprintf(stderr, "Not yet implemented: %s\n", what);
  std::exit(1);
}
