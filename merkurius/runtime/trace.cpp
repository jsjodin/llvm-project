//===-- runtime/trace.cpp - Run time conditions of trace goals ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A trace goal's run_time(env("V")) condition holds if the environment
// variable V exists when the program starts (reference manual, "Trace
// goals"), whatever its value, even empty: main saves a copy of the
// environment before the program runs, so that the program's own changes
// to its environment do not count. Like the rest of the runtime, this uses
// only C's library.
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <cstring>

extern "C" char **environ;

namespace {
char **saved{nullptr};
} // namespace

extern "C" void merk_save_environment(void) {
  std::size_t count{0};
  while (environ && environ[count]) {
    ++count;
  }
  saved = static_cast<char **>(std::malloc((count + 1) * sizeof(char *)));
  if (!saved) {
    return;
  }
  for (std::size_t i{0}; i < count; ++i) {
    saved[i] = strdup(environ[i]);
  }
  saved[count] = nullptr;
}

// Whether the environment variable existed when the program started.
extern "C" int merk_env_at_start(const char *name) {
  std::size_t length{std::strlen(name)};
  for (char **entry = saved; entry && *entry; ++entry) {
    if (std::strncmp(*entry, name, length) == 0 && (*entry)[length] == '=') {
      return 1;
    }
  }
  return 0;
}
