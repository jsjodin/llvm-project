//===-- runtime/frame-info.cpp - Where a frame keeps its return address ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reads the .eh_frame_hdr search table of the object that holds a code
// address, its FDE and CIE, and runs their call frame instructions up to the
// address, for the rule of the canonical frame address (DWARF 5, 6.4; the
// Linux Standard Base's .eh_frame and .eh_frame_hdr).
//
//===----------------------------------------------------------------------===//

#include "frame-info.h"
#include <cstring>
#include <link.h>

namespace merkurius::runtime {
namespace {

// x86-64's DWARF register numbers.
constexpr std::uint64_t kFramePointer{6};

// Pointer encodings (DW_EH_PE_*).
constexpr std::uint8_t kOmit{0xff};
constexpr std::uint8_t kFormatMask{0x0f};
constexpr std::uint8_t kApplicationMask{0x70};
constexpr std::uint8_t kPcRelative{0x10};
constexpr std::uint8_t kDataRelative{0x30};
constexpr std::uint8_t kSigned4DataRelative{0x3b};

struct Reader {
  const std::uint8_t *at;

  template <typename T> T Fixed() {
    T value;
    std::memcpy(&value, at, sizeof value);
    at += sizeof value;
    return value;
  }
  std::uint64_t Unsigned() {
    std::uint64_t value{0};
    unsigned shift{0};
    std::uint8_t byte;
    do {
      byte = *at++;
      if (shift < 64) {
        value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
      }
      shift += 7;
    } while (byte & 0x80);
    return value;
  }
  std::int64_t Signed() {
    std::int64_t value{0};
    unsigned shift{0};
    std::uint8_t byte;
    do {
      byte = *at++;
      if (shift < 64) {
        value |= static_cast<std::int64_t>(byte & 0x7f) << shift;
      }
      shift += 7;
    } while (byte & 0x80);
    if (shift < 64 && (byte & 0x40)) {
      value |= -(std::int64_t{1} << shift);
    }
    return value;
  }
  // A pointer in `encoding`, relative to `data` if data-relative; false if
  // the encoding is one the compilers do not use here.
  bool Pointer(std::uint8_t encoding, std::uint64_t data, std::uint64_t &out) {
    std::uint64_t base{0};
    switch (encoding & kApplicationMask) {
    case 0:
      break;
    case kPcRelative:
      base = reinterpret_cast<std::uint64_t>(at);
      break;
    case kDataRelative:
      base = data;
      break;
    default:
      return false;
    }
    std::uint64_t value;
    switch (encoding & kFormatMask) {
    case 0x00: // absptr
    case 0x04: // udata8
    case 0x0c: // sdata8
      value = Fixed<std::uint64_t>();
      break;
    case 0x02: // udata2
      value = Fixed<std::uint16_t>();
      break;
    case 0x0a: // sdata2
      value = static_cast<std::uint64_t>(Fixed<std::int16_t>());
      break;
    case 0x03: // udata4
      value = Fixed<std::uint32_t>();
      break;
    case 0x0b: // sdata4
      value = static_cast<std::uint64_t>(Fixed<std::int32_t>());
      break;
    case 0x01: // uleb128
      value = Unsigned();
      break;
    case 0x09: // sleb128
      value = static_cast<std::uint64_t>(Signed());
      break;
    default:
      return false;
    }
    out = base + value;
    return true;
  }
};

struct Search {
  std::uint64_t address;
  const std::uint8_t *header; // .eh_frame_hdr
};

int FindHeader(dl_phdr_info *info, std::size_t, void *data) {
  auto *search{static_cast<Search *>(data)};
  const ElfW(Phdr) * header{nullptr};
  bool holds{false};
  for (ElfW(Half) i{0}; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr) & segment{info->dlpi_phdr[i]};
    std::uint64_t begin{info->dlpi_addr + segment.p_vaddr};
    if (segment.p_type == PT_LOAD && search->address >= begin &&
        search->address < begin + segment.p_memsz) {
      holds = true;
    } else if (segment.p_type == PT_GNU_EH_FRAME) {
      header = &segment;
    }
  }
  if (!holds || !header) {
    return 0;
  }
  search->header =
      reinterpret_cast<const std::uint8_t *>(info->dlpi_addr + header->p_vaddr);
  return 1;
}

// The FDE whose range holds `address`, from the header's sorted table.
const std::uint8_t *FindFde(const std::uint8_t *header, std::uint64_t address) {
  if (header[0] != 1) {
    return nullptr;
  }
  std::uint8_t frameEncoding{header[1]}, countEncoding{header[2]},
      tableEncoding{header[3]};
  auto data{reinterpret_cast<std::uint64_t>(header)};
  Reader reader{header + 4};
  std::uint64_t frame, count;
  if (!reader.Pointer(frameEncoding, data, frame) || countEncoding == kOmit ||
      !reader.Pointer(countEncoding, data, count) ||
      tableEncoding != kSigned4DataRelative) {
    return nullptr;
  }
  // Pairs of the start of a function and its FDE, each 4 bytes from data.
  const std::uint8_t *table{reader.at};
  std::uint64_t low{0}, high{count};
  while (low < high) {
    std::uint64_t middle{low + (high - low) / 2};
    std::int32_t start;
    std::memcpy(&start, table + 8 * middle, 4);
    if (data + static_cast<std::uint64_t>(std::int64_t{start}) <= address) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  if (low == 0) {
    return nullptr;
  }
  std::int32_t fde;
  std::memcpy(&fde, table + 8 * (low - 1) + 4, 4);
  return reinterpret_cast<const std::uint8_t *>(
      data + static_cast<std::uint64_t>(std::int64_t{fde}));
}

struct Rule {
  std::uint64_t reg;
  std::int64_t offset;
};

struct Cie {
  std::uint64_t codeAlign;
  std::int64_t dataAlign;
  std::uint8_t fdeEncoding;
  bool augmented; // 'z': an FDE has the length of its augmentation data
  const std::uint8_t *instructions;
  const std::uint8_t *end;
};

bool ReadCie(const std::uint8_t *at, Cie &cie) {
  Reader reader{at};
  std::uint32_t length{reader.Fixed<std::uint32_t>()};
  if (length == 0xffffffff) {
    return false;
  }
  cie.end = reader.at + length;
  if (reader.Fixed<std::uint32_t>() != 0) {
    return false;
  }
  std::uint8_t version{reader.Fixed<std::uint8_t>()};
  const char *augmentation{reinterpret_cast<const char *>(reader.at)};
  reader.at += std::strlen(augmentation) + 1;
  if (augmentation[0] == 'e' && augmentation[1] == 'h') {
    reader.at += sizeof(void *);
  }
  cie.codeAlign = reader.Unsigned();
  cie.dataAlign = reader.Signed();
  if (version == 1) {
    reader.at += 1;
  } else {
    reader.Unsigned();
  }
  cie.fdeEncoding = 0;
  cie.augmented = augmentation[0] == 'z';
  if (cie.augmented) {
    std::uint64_t size{reader.Unsigned()};
    const std::uint8_t *data{reader.at};
    for (const char *c{augmentation + 1}; *c; ++c) {
      if (*c == 'R') {
        cie.fdeEncoding = reader.Fixed<std::uint8_t>();
      } else if (*c == 'P') {
        std::uint8_t encoding{reader.Fixed<std::uint8_t>()};
        std::uint64_t ignored;
        if (!reader.Pointer(encoding & 0x7f, 0, ignored)) {
          return false;
        }
      } else if (*c == 'L') {
        reader.at += 1;
      } else if (*c != 'S' && *c != 'B') {
        return false;
      }
    }
    reader.at = data + size;
  }
  cie.instructions = reader.at;
  return true;
}

// Runs the instructions from `at` to `end` while the location is not past
// `target`, updating `rule`; false on an instruction it does not know.
bool Run(const Cie &cie, const std::uint8_t *at, const std::uint8_t *end,
    std::uint64_t location, std::uint64_t target, Rule &rule) {
  constexpr int kStates{8};
  Rule saved[kStates];
  int depth{0};
  Reader reader{at};
  while (reader.at < end) {
    std::uint8_t op{reader.Fixed<std::uint8_t>()};
    std::uint64_t advance{0};
    switch (op & 0xc0) {
    case 0x40: // DW_CFA_advance_loc
      advance = (op & 0x3f) * cie.codeAlign;
      break;
    case 0x80: // DW_CFA_offset
      reader.Unsigned();
      continue;
    case 0xc0: // DW_CFA_restore
      continue;
    default:
      break;
    }
    if ((op & 0xc0) == 0) {
      switch (op) {
      case 0x00: // nop
        continue;
      case 0x01: { // set_loc
        std::uint64_t to;
        if (!reader.Pointer(cie.fdeEncoding, 0, to)) {
          return false;
        }
        if (to > target) {
          return true;
        }
        location = to;
        continue;
      }
      case 0x02:
        advance = reader.Fixed<std::uint8_t>() * cie.codeAlign;
        break;
      case 0x03:
        advance = reader.Fixed<std::uint16_t>() * cie.codeAlign;
        break;
      case 0x04:
        advance = reader.Fixed<std::uint32_t>() * cie.codeAlign;
        break;
      case 0x05: // offset_extended
      case 0x09: // register
        reader.Unsigned();
        reader.Unsigned();
        continue;
      case 0x06: // restore_extended
      case 0x07: // undefined
      case 0x08: // same_value
        reader.Unsigned();
        continue;
      case 0x0a: // remember_state
        if (depth == kStates) {
          return false;
        }
        saved[depth++] = rule;
        continue;
      case 0x0b: // restore_state
        if (depth == 0) {
          return false;
        }
        rule = saved[--depth];
        continue;
      case 0x0c: // def_cfa
        rule.reg = reader.Unsigned();
        rule.offset = static_cast<std::int64_t>(reader.Unsigned());
        continue;
      case 0x0d: // def_cfa_register
        rule.reg = reader.Unsigned();
        continue;
      case 0x0e: // def_cfa_offset
        rule.offset = static_cast<std::int64_t>(reader.Unsigned());
        continue;
      case 0x10: // expression
      case 0x16: { // val_expression
        reader.Unsigned();
        std::uint64_t size{reader.Unsigned()};
        reader.at += size;
        continue;
      }
      case 0x11: // offset_extended_sf
      case 0x15: // val_offset_sf
        reader.Unsigned();
        reader.Signed();
        continue;
      case 0x12: // def_cfa_sf
        rule.reg = reader.Unsigned();
        rule.offset = reader.Signed() * cie.dataAlign;
        continue;
      case 0x13: // def_cfa_offset_sf
        rule.offset = reader.Signed() * cie.dataAlign;
        continue;
      case 0x14: // val_offset
        reader.Unsigned();
        reader.Unsigned();
        continue;
      case 0x2e: // GNU_args_size
        reader.Unsigned();
        continue;
      default: // def_cfa_expression and anything unknown
        return false;
      }
    }
    if (location + advance > target) {
      return true;
    }
    location += advance;
  }
  return true;
}

} // namespace

std::int64_t ReserveAt(std::uint64_t returnAddress) {
  // The rule during the call, at the call instruction's last byte.
  std::uint64_t target{returnAddress - 1};
  Search search{target, nullptr};
  if (!dl_iterate_phdr(FindHeader, &search)) {
    return -1;
  }
  const std::uint8_t *fde{FindFde(search.header, target)};
  if (!fde) {
    return -1;
  }
  Reader reader{fde};
  std::uint32_t length{reader.Fixed<std::uint32_t>()};
  if (length == 0 || length == 0xffffffff) {
    return -1;
  }
  const std::uint8_t *end{reader.at + length};
  const std::uint8_t *idField{reader.at};
  std::uint32_t id{reader.Fixed<std::uint32_t>()};
  Cie cie;
  if (id == 0 || !ReadCie(idField - id, cie)) {
    return -1;
  }
  std::uint64_t begin, range;
  if (!reader.Pointer(cie.fdeEncoding, 0, begin) ||
      !reader.Pointer(cie.fdeEncoding & kFormatMask, 0, range) ||
      target < begin || target >= begin + range) {
    return -1;
  }
  if (cie.augmented) {
    std::uint64_t size{reader.Unsigned()};
    reader.at += size;
  }
  Rule rule{0, 0};
  if (!Run(cie, cie.instructions, cie.end, begin, ~std::uint64_t{0}, rule) ||
      !Run(cie, reader.at, end, begin, target, rule) ||
      rule.reg != kFramePointer) {
    return -1;
  }
  // The canonical frame address is above the return address and the saved
  // frame pointer, which the frame pointer points to.
  return rule.offset - 16;
}

} // namespace merkurius::runtime
