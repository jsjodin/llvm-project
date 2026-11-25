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
#include "llvm/Support/Unicode.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <utility>

namespace merkurius::parser {

AllSources::AllSources() : range_{1, 1} {
  // Start the origin_ array with a dummy entry that has a forced provenance,
  // so that provenance offset 0 remains reserved as an uninitialized
  // value.
  origin_.push_back({nullptr, range_});
}

AllSources::~AllSources() {}

void AllSources::AppendSearchPathDirectory(std::string directory) {
  searchPath_.push_back(directory);
}

SourceFile *AllSources::AddSourceFile(
    std::unique_ptr<SourceFile> source) {
  ProvenanceRange covers{range_.NextAfter(), source->bytes() + 1};
  CHECK(range_.AnnexIfPredecessor(covers));
  CHECK(origin_.back().covers.ImmediatelyPrecedes(covers));
  fileOrigins_.emplace(source->content().data(), origin_.size());
  origin_.push_back({source.get(), covers});
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
    const std::string &prefix, bool echoSourceLine) const {
  const Origin *origin{nullptr};
  if (range) {
    CHECK(IsValid(*range));
    origin = &MapToOrigin(range->start());
  }
  if (!origin || !origin->source) {
    o << prefix << message << '\n';
    return;
  }
  const SourceFile *source{origin->source};
  std::size_t offset{origin->covers.MemberOffset(range->start())};
  SourcePosition pos{source->GetSourcePosition(offset)};
  o << pos.path << ':' << pos.line << ':' << pos.column << ": " << prefix
    << message << '\n';
  if (!echoSourceLine) {
    return;
  }
  llvm::StringRef line{source->GetLine(pos.trueLineNumber)};
  std::size_t column{std::min<std::size_t>(pos.column - 1, line.size())};
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

std::optional<SourcePosition> AllSources::GetSourcePosition(
    Provenance prov) const {
  const Origin &origin{MapToOrigin(prov)};
  if (!origin.source) {
    return std::nullopt;
  }
  return origin.source->GetSourcePosition(origin.covers.MemberOffset(prov));
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
  llvm::StringRef content{origin.source->content()};
  if (block.end() > content.end()) {
    return std::nullopt;
  }
  // An empty block, such as the eof token's spelling, is a position: one
  // provenance, which at the end of a file is the file's end provenance.
  return ProvenanceRange{origin.covers.start() +
          static_cast<std::size_t>(block.begin() - content.begin()),
      std::max<std::size_t>(block.size(), 1)};
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
    if (m.source) {
      o << "file " << m.source->path();
      sources.push_back(m.source);
    } else {
      o << "reserved";
    }
    o << '\n';
  }
  for (const SourceFile *sf : sources) {
    sf->Dump(o);
  }
  return o;
}

} // namespace merkurius::parser
