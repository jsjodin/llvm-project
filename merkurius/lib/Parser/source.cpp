//===-- lib/Parser/source.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Copied from llvm-project flang/lib/Parser/source.cpp
// at upstream commit d81bd0216be61d9689fc04e74272b73bc6d8a20b,
// and modified for Merkurius.
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/source.h"
#include "merkurius/Common/idioms.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <string>
#include <vector>

namespace merkurius::parser {

SourceFile::~SourceFile() { Close(); }

// Every file has at least one line, possibly empty, so that every offset
// up to and including the end of the file has a position. A newline that
// ends the file does not start another line.
void SourceFile::RecordLineStarts() {
  origins_.emplace(1, SourcePositionOrigin{path_, 1});
  llvm::StringRef text{content()};
  std::size_t at{0};
  do { // "at" is always at the beginning of a source line
    lineStart_.push_back(at);
    std::size_t newline{text.find('\n', at)};
    at = newline == llvm::StringRef::npos ? text.size() : newline + 1;
  } while (at < text.size());
  lineStart_.shrink_to_fit();
}

// Check for a Unicode byte order mark (BOM).
void SourceFile::IdentifyPayload() {
  constexpr llvm::StringLiteral UTF8_BOM{"\xef\xbb\xbf"};
  if (buf_->getBuffer().starts_with(UTF8_BOM)) {
    bom_end_ = UTF8_BOM.size();
  }
}

std::string DirectoryName(std::string path) {
  llvm::SmallString<128> pathBuf{path};
  llvm::sys::path::remove_filename(pathBuf);
  return pathBuf.str().str();
}

std::optional<std::string> LocateSourceFile(
    std::string name, const std::list<std::string> &searchPath) {
  if (name == "-" || llvm::sys::path::is_absolute(name)) {
    return name;
  }
  for (const std::string &dir : searchPath) {
    llvm::SmallString<128> path;
    // If the file is found in the current directory, don't append the
    // directory path. This preserves the user's format.
    if (dir == ".") {
      path = name;
    } else {
      path = dir;
      llvm::sys::path::append(path, name);
    }
    bool isDir{false};
    auto er = llvm::sys::fs::is_directory(path, isDir);
    if (!er && !isDir) {
      return path.str().str();
    }
  }
  return std::nullopt;
}

bool SourceFile::Open(std::string path, llvm::raw_ostream &error) {
  Close();
  auto bufOr{llvm::MemoryBuffer::getFile(path)};
  if (!bufOr) {
    error << bufOr.getError().message();
    return false;
  }
  SetBuffer(std::move(path), std::move(bufOr.get()));
  return true;
}

bool SourceFile::ReadStandardInput(llvm::raw_ostream &error) {
  Close();
  auto bufOr{llvm::MemoryBuffer::getSTDIN()};
  if (!bufOr) {
    error << bufOr.getError().message();
    return false;
  }
  SetBuffer("<stdin>", std::move(bufOr.get()));
  return true;
}

void SourceFile::SetBuffer(
    std::string path, std::unique_ptr<llvm::MemoryBuffer> buffer) {
  Close();
  path_ = std::move(path);
  buf_ = std::move(buffer);
  IdentifyPayload();
  RecordLineStarts();
}

void SourceFile::Close() {
  path_.clear();
  buf_.reset();
  lineStart_.clear();
  bom_end_ = 0;
  distinctPaths_.clear();
  origins_.clear();
}

SourcePosition SourceFile::GetSourcePosition(std::size_t at) const {
  CHECK(at <= bytes());
  auto it{llvm::upper_bound(lineStart_, at)};
  auto trueLineNumber{std::distance(lineStart_.begin(), it - 1) + 1};
  auto ub{origins_.upper_bound(trueLineNumber)};
  auto column{static_cast<int>(at - lineStart_[trueLineNumber - 1] + 1)};
  if (ub == origins_.begin()) {
    return {*this, path_, static_cast<int>(trueLineNumber), column,
        static_cast<int>(trueLineNumber)};
  } else {
    --ub;
    const SourcePositionOrigin &origin{ub->second};
    auto lineNumber{
        trueLineNumber - ub->first + static_cast<std::size_t>(origin.line)};
    return {*this, origin.path, static_cast<int>(lineNumber), column,
        static_cast<int>(trueLineNumber)};
  }
}

llvm::StringRef SourceFile::GetLine(int trueLineNumber) const {
  llvm::StringRef line{
      content().drop_front(GetLineStartOffset(trueLineNumber))};
  line = line.take_until([](char ch) { return ch == '\n'; });
  if (line.ends_with("\r")) {
    line = line.drop_back();
  }
  return line;
}

const std::string &SourceFile::SavePath(std::string &&path) {
  return *distinctPaths_.emplace(std::move(path)).first;
}

void SourceFile::LineDirective(
    int trueLineNumber, const std::string &path, int lineNumber) {
  // A later directive for the same line replaces an earlier one.
  origins_.erase(trueLineNumber);
  origins_.emplace(trueLineNumber, SourcePositionOrigin{path, lineNumber});
}

llvm::raw_ostream &SourceFile::Dump(llvm::raw_ostream &o) const {
  o << "SourceFile '" << path_ << "': " << bytes() << " bytes, " << lines()
    << (lines() == 1 ? " line" : " lines")
    << (hasByteOrderMark() ? ", byte order mark" : "") << '\n';
  for (const auto &[at, spo] : origins_) {
    o << "  origin_[" << at << "] -> '" << spo.path << "' " << spo.line << '\n';
  }
  return o;
}
} // namespace merkurius::parser
