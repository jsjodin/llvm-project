//===------- Offload API tests - olCreateQueueWithComputeUnitMask ---------===//
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

struct olCreateQueueWithComputeUnitMaskTest : OffloadDeviceTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::SetUp());
    uint32_t Units = 0;
    ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_NUM_COMPUTE_UNITS,
                                   sizeof(Units), &Units));
    NumUnits = Units;
    DeviceMask.assign((Units + 31) / 32, 0);
    ASSERT_SUCCESS_OR_UNSUPPORTED(olGetDeviceComputeUnitMask(
        Device, DeviceMask.size(), DeviceMask.data()));
  }

  // The first half of the device's compute units, in pairs, as processors
  // with work group processors enable them.
  std::vector<uint32_t> halfMask() const {
    uint32_t Set = 0;
    for (uint32_t Word : DeviceMask)
      Set += __builtin_popcount(Word);
    uint32_t Wanted = std::max(2u, Set / 2 & ~1u);
    std::vector<uint32_t> Mask(DeviceMask.size(), 0);
    for (uint32_t Bit = 0; Bit < NumUnits && Wanted > 0; ++Bit) {
      if ((DeviceMask[Bit / 32] >> (Bit % 32)) & 1) {
        Mask[Bit / 32] |= 1u << (Bit % 32);
        --Wanted;
      }
    }
    return Mask;
  }

  uint32_t NumUnits = 0;
  std::vector<uint32_t> DeviceMask;
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olCreateQueueWithComputeUnitMaskTest);

TEST_P(olCreateQueueWithComputeUnitMaskTest, Success) {
  ol_queue_handle_t Queue = nullptr;
  ASSERT_SUCCESS(olCreateQueueWithComputeUnitMask(
      Context, Device, DeviceMask.data(), DeviceMask.size(), &Queue));
  ASSERT_NE(Queue, nullptr);
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_SUCCESS(olDestroyQueue(Queue));
}

TEST_P(olCreateQueueWithComputeUnitMaskTest, SuccessHalf) {
  std::vector<uint32_t> Mask = halfMask();
  ol_queue_handle_t Queue = nullptr;
  ASSERT_SUCCESS(olCreateQueueWithComputeUnitMask(Context, Device, Mask.data(),
                                                  Mask.size(), &Queue));
  uint32_t Value = 0;
  ASSERT_SUCCESS(olLaunchHostFunction(
      Queue, [](void *Data) { *static_cast<uint32_t *>(Data) = 1; }, &Value));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  EXPECT_EQ(Value, 1u);
  ASSERT_SUCCESS(olDestroyQueue(Queue));
}

// Queues limited to compute units are each their own, and a queue destroyed
// while busy is released once it is done, never taken by another queue.
TEST_P(olCreateQueueWithComputeUnitMaskTest, SuccessDestroyedBusy) {
  SKIP_IF_FORCE_SYNC_OPS();
  std::vector<uint32_t> Mask = halfMask();
  for (int I = 0; I < 64; ++I) {
    ManuallyTriggeredTask Manual;
    ol_queue_handle_t Limited = nullptr;
    ASSERT_SUCCESS(olCreateQueueWithComputeUnitMask(
        Context, Device, Mask.data(), Mask.size(), &Limited));
    ASSERT_SUCCESS(Manual.enqueue(Limited));
    ASSERT_SUCCESS(olDestroyQueue(Limited));
    ASSERT_SUCCESS(Manual.trigger());
    ol_queue_handle_t Plain = nullptr;
    ASSERT_SUCCESS(olCreateQueue(Context, Device, &Plain));
    ASSERT_SUCCESS(olSyncQueue(Plain));
    ASSERT_SUCCESS(olDestroyQueue(Plain));
  }
}

// The kernels of two queues of one mask run at once: each waits for the
// other's flag, and gives up and reports it if they run one after the other.
// The kernel uses no scratch: on AMDGPU, a kernel that does may run alone.
TEST_P(olCreateQueueWithComputeUnitMaskTest, SuccessSameMaskConcurrent) {
  std::unique_ptr<llvm::MemoryBuffer> Bin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("wait_for_peer", Device, Bin));
  ol_program_handle_t Program = nullptr;
  ASSERT_SUCCESS(olCreateProgram(Context, Device, Bin->getBufferStart(),
                                 Bin->getBufferSize(), &Program));
  ol_symbol_handle_t Kernel = nullptr;
  ASSERT_SUCCESS(
      olGetSymbol(Program, "wait_for_peer", OL_SYMBOL_KIND_KERNEL, &Kernel));

  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_MANAGED,
                            3 * sizeof(uint32_t), &Mem));
  uint32_t *Flags = static_cast<uint32_t *>(Mem);
  uint32_t *TimedOut = Flags + 2;
  Flags[0] = Flags[1] = *TimedOut = 0;

  std::vector<uint32_t> Mask = halfMask();
  ol_queue_handle_t Queues[2] = {nullptr, nullptr};
  uint32_t Mine[2] = {0, 1};
  ol_kernel_launch_size_args_t Size{};
  Size.Dimensions = 1;
  Size.NumGroups = {1, 1, 1};
  Size.GroupSize = {1, 1, 1};
  for (int I = 0; I < 2; ++I) {
    ASSERT_SUCCESS(olCreateQueueWithComputeUnitMask(
        Context, Device, Mask.data(), Mask.size(), &Queues[I]));
    void *ArgPtrs[] = {&Flags, &Mine[I], &TimedOut};
    size_t ArgSizes[] = {sizeof(Flags), sizeof(Mine[I]), sizeof(TimedOut)};
    ASSERT_SUCCESS(olLaunchKernel(Queues[I], Device, Kernel, &Size, nullptr,
                                  std::size(ArgPtrs), ArgPtrs, ArgSizes));
  }
  for (ol_queue_handle_t Queue : Queues) {
    ASSERT_SUCCESS(olSyncQueue(Queue));
    ASSERT_SUCCESS(olDestroyQueue(Queue));
  }
  EXPECT_EQ(*TimedOut, 0u);
  EXPECT_EQ(Flags[0], 1u);
  EXPECT_EQ(Flags[1], 1u);

  ASSERT_SUCCESS(olMemFree(Context, Mem));
  ASSERT_SUCCESS(olDestroyProgram(Program));
}

TEST_P(olCreateQueueWithComputeUnitMaskTest, InvalidNoUnits) {
  std::vector<uint32_t> Mask(DeviceMask.size(), 0);
  ol_queue_handle_t Queue = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_ARGUMENT,
               olCreateQueueWithComputeUnitMask(Context, Device, Mask.data(),
                                                Mask.size(), &Queue));
}

TEST_P(olCreateQueueWithComputeUnitMaskTest, InvalidUnitOutsideDevice) {
  std::vector<uint32_t> Mask(DeviceMask);
  Mask.push_back(1);
  ol_queue_handle_t Queue = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_ARGUMENT,
               olCreateQueueWithComputeUnitMask(Context, Device, Mask.data(),
                                                Mask.size(), &Queue));
}

TEST_P(olCreateQueueWithComputeUnitMaskTest, InvalidNullPointerMask) {
  ol_queue_handle_t Queue = nullptr;
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_POINTER,
      olCreateQueueWithComputeUnitMask(Context, Device, nullptr, 1, &Queue));
}

TEST_P(olCreateQueueWithComputeUnitMaskTest, InvalidNullHandleDevice) {
  ol_queue_handle_t Queue = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olCreateQueueWithComputeUnitMask(Context, nullptr,
                                                DeviceMask.data(),
                                                DeviceMask.size(), &Queue));
}
