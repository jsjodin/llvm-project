//===-- include/merkurius/Parser/source.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Copied from llvm-project flang/include/flang/Parser/source.h
// at upstream commit d81bd0216be61d9689fc04e74272b73bc6d8a20b,
// and modified for Merkurius.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_PARSER_SOURCE_H_
#define MERKURIUS_PARSER_SOURCE_H_

// Source file content is kept exactly as read, so that printing the tokens
// of a file with their trivia gives back the file byte for byte. Unlike
// Flang, line endings are not normalized and no final newline is added: a
// carriage return is whitespace to Mercury. Source is UTF-8; a Unicode byte
// order mark is recognized and kept out of content().

#include "merkurius/Common/reference.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace merkurius::parser {

std::string DirectoryName(std::string path);
std::optional<std::string> LocateSourceFile(
    std::string name, const std::list<std::string> &searchPath);

class SourceFile;

struct SourcePosition {
  common::Reference<const SourceFile> sourceFile;
  common::Reference<const std::string>
      path; // may not be sourceFile.path() after pragma source_file
  int line, column; // column counts bytes from 1
  int trueLineNumber;
};

class SourceFile {
public:
  SourceFile() {}
  ~SourceFile();
  const std::string &path() const { return path_; }
  // The file's bytes, without any byte order mark.
  llvm::StringRef content() const {
    return buf_->getBuffer().drop_front(bom_end_);
  }
  bool hasByteOrderMark() const { return bom_end_ > 0; }
  std::size_t bytes() const { return content().size(); }
  std::size_t lines() const { return lineStart_.size(); }

  bool Open(std::string path, llvm::raw_ostream &error);
  bool ReadStandardInput(llvm::raw_ostream &error);
  // Takes the contents from a buffer already in memory.
  void SetBuffer(std::string path, std::unique_ptr<llvm::MemoryBuffer>);
  void Close();

  // Valid for 0 <= offset <= bytes(); bytes() is the end of the file.
  SourcePosition GetSourcePosition(std::size_t offset) const;
  std::size_t GetLineStartOffset(int trueLineNumber) const {
    return lineStart_.at(trueLineNumber - 1);
  }
  // The text of a line, without its line ending.
  llvm::StringRef GetLine(int trueLineNumber) const;

  const std::string &SavePath(std::string &&);
  // From line trueLineNumber on, positions are reported as lineNumber in
  // path, counting up from there. Used for '#' line number directives and
  // pragma source_file.
  void LineDirective(int trueLineNumber, const std::string &, int lineNumber);
  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;

private:
  struct SourcePositionOrigin {
    const std::string &path;
    int line;
  };

  void IdentifyPayload();
  void RecordLineStarts();

  std::string path_;
  std::unique_ptr<llvm::MemoryBuffer> buf_;
  std::vector<std::size_t> lineStart_;
  std::size_t bom_end_{0};
  std::set<std::string> distinctPaths_;
  std::map<std::size_t, SourcePositionOrigin> origins_;
};
} // namespace merkurius::parser
#endif // MERKURIUS_PARSER_SOURCE_H_
