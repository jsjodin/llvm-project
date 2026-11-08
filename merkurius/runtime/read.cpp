//===-- runtime/read.cpp - Reading the standard input ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What library/io.m reads from the standard input, with C's stdio, as mmc's
// programs read it: a line is its bytes up to and with its newline, which
// the last line may lack, kept as they are even when they are not UTF-8.
//
//===----------------------------------------------------------------------===//

#include "process.h"

#include <poll.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" char *merk_string_new(std::int64_t bytes);

namespace {

// What merk_read_line found.
enum ReadLine : std::int64_t {
  kLine = 0,
  kEndOfFile = 1,
  // A NUL, which no string may hold: the line up to it is dropped, and the
  // next read starts after it, as in mmc.
  kNull = 2,
  kError = 3,
};

// Before a character is read: if that would wait for input, the process waits
// on a thread of the blocking pool, and holds up no worker. What is in the
// stdio buffer, or ready to be read, costs no move.
void MoveIfInputWouldWait(bool &moved) {
#ifdef __GLIBC__
  if (moved || stdin->_IO_read_ptr < stdin->_IO_read_end) {
    return;
  }
#else
  if (moved) {
    return;
  }
#endif
  pollfd input{fileno(stdin), POLLIN, 0};
  if (poll(&input, 1, 0) == 0) {
    merkurius::runtime::Pin();
    moved = true;
  }
}

std::int64_t ReadLine(char **line, bool &moved);

} // namespace

extern "C" std::int64_t merk_read_line(char **line) {
  bool moved{false};
  std::int64_t result{ReadLine(line, moved)};
  if (moved) {
    merkurius::runtime::Unpin();
  }
  return result;
}

namespace {
// The next line of the standard input in `*line`, a new string, for kLine;
// for kError, the C library's message in `*line`.
std::int64_t ReadLine(char **line, bool &moved) {
  std::size_t size{0}, capacity{128};
  char *buffer{static_cast<char *>(std::malloc(capacity))};
  int c;
  for (;;) {
    MoveIfInputWouldWait(moved);
    c = std::getc(stdin);
    if (c == EOF) {
      break;
    }
    if (c == 0) {
      std::free(buffer);
      return kNull;
    }
    if (size == capacity) {
      capacity *= 2;
      buffer = static_cast<char *>(std::realloc(buffer, capacity));
    }
    buffer[size++] = static_cast<char>(c);
    if (c == '\n') {
      break;
    }
  }
  if (c == EOF && std::ferror(stdin)) {
    const char *message{std::strerror(errno)};
    std::size_t bytes{std::strlen(message)};
    *line = merk_string_new(static_cast<std::int64_t>(bytes));
    std::memcpy(*line, message, bytes);
    std::free(buffer);
    return kError;
  }
  if (c == EOF && size == 0) {
    std::free(buffer);
    return kEndOfFile;
  }
  *line = merk_string_new(static_cast<std::int64_t>(size));
  std::memcpy(*line, buffer, size);
  std::free(buffer);
  return kLine;
}

} // namespace
