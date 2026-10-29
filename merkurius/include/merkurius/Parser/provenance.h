//===-- include/merkurius/Parser/provenance.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Copied from llvm-project flang/include/flang/Parser/provenance.h
// at upstream commit d81bd0216be61d9689fc04e74272b73bc6d8a20b,
// and modified for Merkurius.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_PARSER_PROVENANCE_H_
#define MERKURIUS_PARSER_PROVENANCE_H_

#include "char-block.h"
#include "source.h"
#include "merkurius/Common/idioms.h"
#include "merkurius/Common/interval.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace merkurius::parser {

// Every character of every source file, and every piece of text the
// compiler inserts, has a provenance: an offset into an (unmaterialized)
// concatenation of all of them. Given a Provenance, we can find the origin
// that contains it in time O(log(# of origins)).
//
// Flang's prescanner rewrites its input into a cooked character stream and
// keeps maps between cooked offsets and provenances. Merkurius has no such
// stage: the lexer reads each source file's bytes directly (see source.h),
// so a CharBlock points into a SourceFile's buffer and AllSources maps it to
// its provenance itself.

class Provenance {
public:
  Provenance() {}
  Provenance(std::size_t offset) : offset_{offset} { CHECK(offset > 0); }
  Provenance(const Provenance &that) = default;
  Provenance(Provenance &&that) = default;
  Provenance &operator=(const Provenance &that) = default;
  Provenance &operator=(Provenance &&that) = default;

  std::size_t offset() const { return offset_; }

  Provenance operator+(ptrdiff_t n) const {
    CHECK(n > -static_cast<ptrdiff_t>(offset_));
    return {offset_ + static_cast<std::size_t>(n)};
  }
  Provenance operator+(std::size_t n) const { return {offset_ + n}; }
  std::size_t operator-(Provenance that) const {
    CHECK(that <= *this);
    return offset_ - that.offset_;
  }
  bool operator<(Provenance that) const { return offset_ < that.offset_; }
  bool operator<=(Provenance that) const { return !(that < *this); }
  bool operator==(Provenance that) const { return offset_ == that.offset_; }
  bool operator!=(Provenance that) const { return !(*this == that); }

private:
  std::size_t offset_{0};
};

using ProvenanceRange = common::Interval<Provenance>;

// A singleton AllSources instance for the whole compilation
// is shared by reference.
class AllSources {
public:
  AllSources();
  ~AllSources();

  std::size_t size() const { return range_.size(); }
  const char &operator[](Provenance) const;

  void ClearSearchPath();
  void AppendSearchPathDirectory(std::string); // new last directory

  // Each of these reads a source file and gives it a range of provenances:
  // one per byte of content(), and one more for its end, so that the end of
  // the file can be reported even when the file is empty. On failure they
  // write the reason to `error` and return null.
  SourceFile *OpenPath(std::string path, llvm::raw_ostream &error);
  SourceFile *Open(std::string path, llvm::raw_ostream &error,
      std::optional<std::string> &&prependPath = std::nullopt);
  SourceFile *ReadStandardInput(llvm::raw_ostream &error);
  SourceFile *AddBuffer(
      std::string path, std::unique_ptr<llvm::MemoryBuffer>);

  // The provenances of a source file opened above, its end included.
  ProvenanceRange GetProvenanceRange(const SourceFile &) const;

  ProvenanceRange AddCompilerInsertion(std::string);
  Provenance CompilerInsertionProvenance(char ch);

  bool IsValid(Provenance at) const { return range_.Contains(at); }
  bool IsValid(ProvenanceRange range) const {
    return range.size() > 0 && range_.Contains(range);
  }
  void setShowColors(bool showColors) { showColors_ = showColors; }
  bool getShowColors() const { return showColors_; }
  void EmitMessage(llvm::raw_ostream &, const std::optional<ProvenanceRange> &,
      const std::string &message, const std::string &prefix,
      llvm::raw_ostream::Colors color, bool echoSourceLine = false) const;
  const SourceFile *GetSourceFile(
      Provenance, std::size_t *offset = nullptr) const;
  const char *GetSource(ProvenanceRange) const;
  std::optional<SourcePosition> GetSourcePosition(Provenance) const;
  std::optional<ProvenanceRange> GetFirstFileProvenance() const;
  std::string GetPath(Provenance) const;
  int GetLineNumber(Provenance) const;

  // Between CharBlocks within source files and their provenances. A block
  // may end at, or be empty at, the end of its file.
  std::optional<ProvenanceRange> GetProvenanceRange(CharBlock) const;
  std::optional<CharBlock> GetCharBlock(ProvenanceRange) const;

  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;

private:
  struct Inclusion {
    const SourceFile &source;
  };
  struct CompilerInsertion {
    std::string text;
  };

  struct Origin {
    Origin(ProvenanceRange, const SourceFile &);
    Origin(ProvenanceRange, const std::string &);

    const char &operator[](std::size_t) const;

    std::variant<Inclusion, CompilerInsertion> u;
    ProvenanceRange covers;
  };

  SourceFile *AddSourceFile(std::unique_ptr<SourceFile>);
  const Origin &MapToOrigin(Provenance) const;

  // Elements are in ascending & contiguous order of .covers.
  std::vector<Origin> origin_;
  ProvenanceRange range_;
  std::map<char, Provenance> compilerInsertionProvenance_;
  std::vector<std::unique_ptr<SourceFile>> ownedSourceFiles_;
  // Index into origin_ of each source file, by the address of its content.
  std::map<const char *, std::size_t> fileOrigins_;
  std::list<std::string> searchPath_;
  bool showColors_{false};
};

} // namespace merkurius::parser
#endif // MERKURIUS_PARSER_PROVENANCE_H_
