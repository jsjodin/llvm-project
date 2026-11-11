//===------- Offload API tests - olDestroyQueue ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olDestroyQueueTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olDestroyQueueTest);

TEST_P(olDestroyQueueTest, Success) {
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  Queue = nullptr;
}

TEST_P(olDestroyQueueTest, SuccessDelayedResolution) {
  ManuallyTriggeredTask Manual;
  SKIP_IF_FORCE_SYNC_OPS();
  ASSERT_SUCCESS(Manual.enqueue(Queue));
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  Queue = nullptr;

  ASSERT_SUCCESS(Manual.trigger());
}

// A queue destroyed while it still has work is taken by the next queue
// created once the work is done, rather than kept until the context is
// destroyed. Only a backend that reports its queues' native handles shows it.
TEST_P(olDestroyQueueTest, SuccessReusedWhenComplete) {
  ManuallyTriggeredTask Manual;
  SKIP_IF_FORCE_SYNC_OPS();
  void *Handle = nullptr;
  ASSERT_SUCCESS_OR_UNSUPPORTED(olGetQueueInfo(
      Queue, OL_QUEUE_INFO_NATIVE_HANDLE, sizeof(Handle), &Handle));
  ASSERT_SUCCESS(Manual.enqueue(Queue));
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  Queue = nullptr;
  ASSERT_SUCCESS(Manual.trigger());

  ASSERT_SUCCESS(olCreateQueue(Context, Device, &Queue));
  void *Reused = nullptr;
  ASSERT_SUCCESS(olGetQueueInfo(Queue, OL_QUEUE_INFO_NATIVE_HANDLE,
                                sizeof(Reused), &Reused));
  ASSERT_EQ(Reused, Handle);
}

TEST_P(olDestroyQueueTest, InvalidNullHandle) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olDestroyQueue(nullptr));
}
