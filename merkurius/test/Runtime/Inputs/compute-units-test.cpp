// The driver of compute-units.test: the units of GPUs' processes, printed as
// hex masks of units 0-95.
#include "../../../runtime/compute-units.h"
#include <cstdint>
#include <cstdio>

using namespace merkurius::runtime;

namespace {

void Units(const char *what, const std::uint32_t *usable, int words) {
  UnitMask mask;
  if (ProcessUnits(usable, words, mask)) {
    std::printf("%s: %08x %08x %08x (%d)", what, mask.words[2], mask.words[1],
        mask.words[0], UnitCount(mask));
    const int percents[]{50, 1};
    for (int percent : percents) {
      UnitMask share{mask};
      KeepShare(share, percent);
      std::printf(", %d%% %08x %08x %08x", percent, share.words[2],
          share.words[1], share.words[0]);
    }
    std::printf("\n");
  } else {
    std::printf("%s: none\n", what);
  }
}

} // namespace

int main() {
  const std::uint32_t all[]{0xffffffff, 0xffffffff, 0xffff};
  Units("80 units", all, 3);
  const std::uint32_t half[]{0xffffffff, 0xff, 0};
  Units("units 0-39", half, 3);
  const std::uint32_t odd[]{0x7e};
  Units("units 1-6", odd, 1);
  const std::uint32_t lone[]{0x15};
  Units("units 0, 2 and 4", lone, 1);
  const std::uint32_t big[]{3, 0, 0, 0, 0, 0, 0, 0, 1};
  Units("a unit beyond the 256th", big, 9);
  return 0;
}
