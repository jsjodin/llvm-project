//===------- Offload API tests - olGetDeviceComputeUnitMask ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>
#include <vector>

using olGetDeviceComputeUnitMaskTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetDeviceComputeUnitMaskTest);

// The mask sets bits only of the device's compute units, at least one, and
// clears the words after them.
TEST_P(olGetDeviceComputeUnitMaskTest, Success) {
  uint32_t Units = 0;
  ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_NUM_COMPUTE_UNITS,
                                 sizeof(Units), &Units));
  size_t Words = (Units + 31) / 32;
  std::vector<uint32_t> Mask(Words + 2, 0xffffffffu);
  ASSERT_SUCCESS_OR_UNSUPPORTED(
      olGetDeviceComputeUnitMask(Device, Mask.size(), Mask.data()));
  uint32_t Set = 0;
  for (size_t Bit = 0; Bit < Mask.size() * 32; ++Bit) {
    if ((Mask[Bit / 32] >> (Bit % 32)) & 1) {
      EXPECT_LT(Bit, Units);
      ++Set;
    }
  }
  EXPECT_GT(Set, 0u);
}

TEST_P(olGetDeviceComputeUnitMaskTest, InvalidSize) {
  uint32_t Units = 0;
  ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_NUM_COMPUTE_UNITS,
                                 sizeof(Units), &Units));
  size_t Words = (Units + 31) / 32;
  std::vector<uint32_t> Mask(Words);
  ASSERT_SUCCESS_OR_UNSUPPORTED(
      olGetDeviceComputeUnitMask(Device, Mask.size(), Mask.data()));
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetDeviceComputeUnitMask(Device, Words - 1, Mask.data()));
}

TEST_P(olGetDeviceComputeUnitMaskTest, InvalidNullHandle) {
  uint32_t Mask[4];
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetDeviceComputeUnitMask(nullptr, 4, Mask));
}

TEST_P(olGetDeviceComputeUnitMaskTest, InvalidNullPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetDeviceComputeUnitMask(Device, 4, nullptr));
}
