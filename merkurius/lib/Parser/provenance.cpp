//===-- lib/Parser/provenance.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Copied from llvm-project flang/lib/Parser/provenance.cpp
// at upstream commit d81bd0216be61d9689fc04e74272b73bc6d8a20b,
// and modified for Merkurius.
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/provenance.h"
#include "merkurius/Common/idioms.h"
#include "merkurius/Common/visit.h"
#include "llvm/Support/Unicode.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <set>
#include <utility>

namespace merkurius::parser {

AllSources::AllSources() : range_{1, 1} {
  // Start the origin_ array with a dummy entry that has a forced provenance,
  // so that provenance offset 0 remains reserved as an uninitialized
  // value.
  origin_.emplace_back(range_, std::string{'?'});
}

AllSources::~AllSources() {}

const char &AllSources::operator[](Provenance at) const {
  const Origin &origin{MapToOrigin(at)};
  return origin[origin.covers.MemberOffset(at)];
}

void AllSources::ClearSearchPath() { searchPath_.clear(); }

void AllSources::AppendSearchPathDirectory(std::string directory) {
  searchPath_.push_back(directory);
}

SourceFile *AllSources::AddSourceFile(
    std::unique_ptr<SourceFile> source) {
  ProvenanceRange covers{range_.NextAfter(), source->bytes() + 1};
  CHECK(range_.AnnexIfPredecessor(covers));
  CHECK(origin_.back().covers.ImmediatelyPrecedes(covers));
  fileOrigins_.emplace(source->content().data(), origin_.size());
  origin_.emplace_back(covers, *source);
  return ownedSourceFiles_.emplace_back(std::move(source)).get();
}

SourceFile *AllSources::OpenPath(
    std::string path, llvm::raw_ostream &error) {
  std::unique_ptr<SourceFile> source{std::make_unique<SourceFile>()};
  if (source->Open(path, error)) {
    return AddSourceFile(std::move(source));
  } else {
    return nullptr;
  }
}

SourceFile *AllSources::Open(std::string path, llvm::raw_ostream &error,
    std::optional<std::string> &&prependPath) {
  if (prependPath) {
    // Set to "." for the initial source file, or to the directory of the
    // importing module for its imports.
    searchPath_.emplace_front(std::move(*prependPath));
  }
  std::optional<std::string> found{LocateSourceFile(path, searchPath_)};
  if (prependPath) {
    searchPath_.pop_front();
  }
  if (found) {
    return OpenPath(*found, error);
  } else {
    error << "source file '" << path << "' was not found";
    return nullptr;
  }
}

SourceFile *AllSources::ReadStandardInput(llvm::raw_ostream &error) {
  std::unique_ptr<SourceFile> source{std::make_unique<SourceFile>()};
  if (source->ReadStandardInput(error)) {
    return AddSourceFile(std::move(source));
  }
  return nullptr;
}

SourceFile *AllSources::AddBuffer(
    std::string path, std::unique_ptr<llvm::MemoryBuffer> buffer) {
  std::unique_ptr<SourceFile> source{std::make_unique<SourceFile>()};
  source->SetBuffer(std::move(path), std::move(buffer));
  return AddSourceFile(std::move(source));
}

ProvenanceRange AllSources::GetProvenanceRange(const SourceFile &source) const {
  auto iter{fileOrigins_.find(source.content().data())};
  CHECK(iter != fileOrigins_.end());
  return origin_[iter->second].covers;
}

ProvenanceRange AllSources::AddCompilerInsertion(std::string text) {
  ProvenanceRange covers{range_.NextAfter(), text.size()};
  CHECK(range_.AnnexIfPredecessor(covers));
  CHECK(origin_.back().covers.ImmediatelyPrecedes(covers));
  origin_.emplace_back(covers, text);
  return covers;
}

static void EmitPrefix(llvm::raw_ostream &o, llvm::raw_ostream::Colors color,
    const std::string &prefix, bool showColors) {
  if (prefix.empty()) {
    return;
  }
  if (showColors) {
    o.changeColor(color, true);
  }
  o << prefix;
  if (showColors) {
    o.resetColor();
  }
}

// Writes whitespace as wide on screen as `text`, keeping its tabs, so that a
// caret written after it lines up under the next character, multi-byte
// UTF-8 characters included.
static void EmitPadding(llvm::raw_ostream &o, llvm::StringRef text) {
  while (!text.empty()) {
    auto [segment, rest]{text.split('\t')};
    int width{llvm::sys::unicode::columnWidthUTF8(segment)};
    o.indent(width >= 0 ? width : static_cast<int>(segment.size()));
    if (segment.size() < text.size()) { // a tab followed the segment
      o << '\t';
    }
    text = rest;
  }
}

void AllSources::EmitMessage(llvm::raw_ostream &o,
    const std::optional<ProvenanceRange> &range, const std::string &message,
    const std::string &prefix, llvm::raw_ostream::Colors color,
    bool echoSourceLine) const {
  if (!range) {
    EmitPrefix(o, color, prefix, this->getShowColors());
    o << message << '\n';
    return;
  }
  CHECK(IsValid(*range));
  const Origin &origin{MapToOrigin(range->start())};
  common::visit(
      common::visitors{
          [&](const Inclusion &inc) {
            std::size_t offset{origin.covers.MemberOffset(range->start())};
            SourcePosition pos{inc.source.GetSourcePosition(offset)};
            o << pos.path << ':' << pos.line << ':' << pos.column << ": ";
            EmitPrefix(o, color, prefix, this->getShowColors());
            o << message << '\n';
            if (echoSourceLine) {
              llvm::StringRef line{inc.source.GetLine(pos.trueLineNumber)};
              std::size_t column{
                  std::min<std::size_t>(pos.column - 1, line.size())};
              o << "  " << line << "\n  ";
              EmitPadding(o, line.take_front(column));
              o << '^';
              if (range->size() > 1) {
                // Underline the rest of the range if it is on this line.
                std::size_t end{column + range->size()};
                if (end <= line.size()) {
                  llvm::StringRef rest{line.slice(column, end)};
                  int width{llvm::sys::unicode::columnWidthUTF8(rest)};
                  if (width < 0) {
                    width = static_cast<int>(rest.size());
                  }
                  for (int j{1}; j < width; ++j) {
                    o << '^';
                  }
                }
              }
              o << '\n';
            }
          },
          [&](const CompilerInsertion &) {
            EmitPrefix(o, color, prefix, this->getShowColors());
            o << message << '\n';
          },
      },
      origin.u);
}

const SourceFile *AllSources::GetSourceFile(
    Provenance at, std::size_t *offset) const {
  const Origin &origin{MapToOrigin(at)};
  return common::visit(common::visitors{
                           [&](const Inclusion &inc) {
                             if (offset) {
                               *offset = origin.covers.MemberOffset(at);
                             }
                             return &inc.source;
                           },
                           [offset](const CompilerInsertion &) {
                             if (offset) {
                               *offset = 0;
                             }
                             return static_cast<const SourceFile *>(nullptr);
                           },
                       },
      origin.u);
}

const char *AllSources::GetSource(ProvenanceRange range) const {
  Provenance start{range.start()};
  const Origin &origin{MapToOrigin(start)};
  return origin.covers.Contains(range)
      ? &origin[origin.covers.MemberOffset(start)]
      : nullptr;
}

std::optional<SourcePosition> AllSources::GetSourcePosition(
    Provenance prov) const {
  const Origin &origin{MapToOrigin(prov)};
  return common::visit(
      common::visitors{
          [&](const Inclusion &inc) -> std::optional<SourcePosition> {
            std::size_t offset{origin.covers.MemberOffset(prov)};
            return inc.source.GetSourcePosition(offset);
          },
          [](const CompilerInsertion &) -> std::optional<SourcePosition> {
            return std::nullopt;
          },
      },
      origin.u);
}

std::optional<ProvenanceRange> AllSources::GetFirstFileProvenance() const {
  for (const auto &origin : origin_) {
    if (std::holds_alternative<Inclusion>(origin.u)) {
      return origin.covers;
    }
  }
  return std::nullopt;
}

std::string AllSources::GetPath(Provenance at) const {
  std::size_t offset{0};
  const SourceFile *source{GetSourceFile(at, &offset)};
  return source ? *source->GetSourcePosition(offset).path : ""s;
}

int AllSources::GetLineNumber(Provenance at) const {
  std::size_t offset{0};
  const SourceFile *source{GetSourceFile(at, &offset)};
  return source ? source->GetSourcePosition(offset).line : 0;
}

Provenance AllSources::CompilerInsertionProvenance(char ch) {
  auto iter{compilerInsertionProvenance_.find(ch)};
  if (iter != compilerInsertionProvenance_.end()) {
    return iter->second;
  }
  ProvenanceRange newCharRange{AddCompilerInsertion(std::string{ch})};
  Provenance newCharProvenance{newCharRange.start()};
  compilerInsertionProvenance_.insert(std::make_pair(ch, newCharProvenance));
  return newCharProvenance;
}

std::optional<ProvenanceRange> AllSources::GetProvenanceRange(
    CharBlock block) const {
  // The file whose content starts at or before the block.
  auto iter{fileOrigins_.upper_bound(block.begin())};
  if (iter == fileOrigins_.begin()) {
    return std::nullopt;
  }
  --iter;
  const Origin &origin{origin_[iter->second]};
  llvm::StringRef content{std::get<Inclusion>(origin.u).source.content()};
  if (block.end() > content.end()) {
    return std::nullopt;
  }
  return ProvenanceRange{origin.covers.start() +
          static_cast<std::size_t>(block.begin() - content.begin()),
      block.size()};
}

std::optional<CharBlock> AllSources::GetCharBlock(ProvenanceRange range) const {
  if (!IsValid(range.start())) {
    return std::nullopt;
  }
  const Origin &origin{MapToOrigin(range.start())};
  const auto *inc{std::get_if<Inclusion>(&origin.u)};
  if (!inc) {
    return std::nullopt;
  }
  llvm::StringRef content{inc->source.content()};
  std::size_t offset{origin.covers.MemberOffset(range.start())};
  if (offset + range.size() > content.size()) {
    return std::nullopt;
  }
  return CharBlock{content.data() + offset, range.size()};
}

AllSources::Origin::Origin(ProvenanceRange r, const SourceFile &source)
    : u{Inclusion{source}}, covers{r} {}
AllSources::Origin::Origin(ProvenanceRange r, const std::string &text)
    : u{CompilerInsertion{text}}, covers{r} {}

const char &AllSources::Origin::operator[](std::size_t n) const {
  // The provenance after a file's last byte stands for its end.
  static const char endOfFile{'\0'};
  return common::visit(common::visitors{
                           [n](const Inclusion &inc) -> const char & {
                             llvm::StringRef content{inc.source.content()};
                             return n < content.size() ? content.data()[n]
                                                       : endOfFile;
                           },
                           [n](const CompilerInsertion &ins) -> const char & {
                             return ins.text[n];
                           },
                       },
      u);
}

const AllSources::Origin &AllSources::MapToOrigin(Provenance at) const {
  CHECK(range_.Contains(at));
  std::size_t low{0}, count{origin_.size()};
  while (count > 1) {
    std::size_t mid{low + (count >> 1)};
    if (at < origin_[mid].covers.start()) {
      count = mid - low;
    } else {
      count -= mid - low;
      low = mid;
    }
  }
  CHECK(origin_[low].covers.Contains(at));
  return origin_[low];
}

static void DumpRange(llvm::raw_ostream &o, const ProvenanceRange &r) {
  o << "[" << r.start().offset() << ".." << r.Last().offset() << "] ("
    << r.size() << " bytes)";
}

llvm::raw_ostream &AllSources::Dump(llvm::raw_ostream &o) const {
  o << "AllSources range_ ";
  DumpRange(o, range_);
  o << '\n';
  std::vector<const SourceFile *> sources;
  for (const Origin &m : origin_) {
    o << "   ";
    DumpRange(o, m.covers);
    o << " -> ";
    common::visit(common::visitors{
                      [&](const Inclusion &inc) {
                        o << "file " << inc.source.path();
                        sources.push_back(&inc.source);
                      },
                      [&](const CompilerInsertion &ins) {
                        o << "compiler '" << ins.text << '\'';
                        if (ins.text.length() == 1) {
                          int ch = ins.text[0];
                          o << "(0x";
                          o.write_hex(ch & 0xff) << ")";
                        }
                      },
                  },
        m.u);
    o << '\n';
  }
  for (const SourceFile *sf : sources) {
    sf->Dump(o);
  }
  return o;
}

} // namespace merkurius::parser
