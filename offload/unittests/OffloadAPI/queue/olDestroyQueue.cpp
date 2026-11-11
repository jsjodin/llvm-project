//===------- Offload API tests - olDestroyQueue ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <atomic>
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

// Destroying a queue does not wait for its pending work and does not cancel
// it: work submitted before the destroy still runs to completion.
TEST_P(olDestroyQueueTest, SuccessPendingWorkCompletes) {
  SKIP_IF_FORCE_SYNC_OPS();

  // Ran is set by a host function queued behind the blocked task.
  std::atomic<bool> Ran{false};
  ManuallyTriggeredTask Task;
  ASSERT_SUCCESS(Task.enqueue(Queue));
  ASSERT_SUCCESS(olLaunchHostFunction(
      Queue,
      [](void *Data) { static_cast<std::atomic<bool> *>(Data)->store(true); },
      &Ran));
  ol_event_handle_t DoneEvent = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &DoneEvent));

  ASSERT_SUCCESS(olDestroyQueue(Queue));
  Queue = nullptr;
  EXPECT_FALSE(Ran.load());

  ASSERT_SUCCESS(Task.trigger());
  ASSERT_SUCCESS(olSyncEvent(DoneEvent));
  EXPECT_TRUE(Ran.load());
  ASSERT_SUCCESS(olDestroyEvent(DoneEvent));
  ASSERT_SUCCESS(olDestroyEvent(Task.CompleteEvent));
}

// An event of a queue that was idle when destroyed stays valid for
// olSyncEvent, olWaitEvents and olDestroyEvent.
TEST_P(olDestroyQueueTest, SuccessEventValidAfterIdleDestroy) {
  ol_queue_handle_t Other = nullptr;
  ASSERT_SUCCESS(olCreateQueue(Context, Device, &Other));

  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olLaunchHostFunction(Queue, [](void *) {}, nullptr));
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  Queue = nullptr;

  ASSERT_SUCCESS(olSyncEvent(Event));
  ASSERT_SUCCESS(olWaitEvents(Other, &Event, 1));
  ASSERT_SUCCESS(olSyncQueue(Other));
  ASSERT_SUCCESS(olDestroyEvent(Event));
  ASSERT_SUCCESS(olDestroyQueue(Other));
}

// An event of a queue that was destroyed with pending work stays valid for
// olSyncEvent, olWaitEvents and olDestroyEvent.
TEST_P(olDestroyQueueTest, SuccessEventValidAfterBusyDestroy) {
  SKIP_IF_FORCE_SYNC_OPS();

  ol_queue_handle_t Other = nullptr;
  ASSERT_SUCCESS(olCreateQueue(Context, Device, &Other));

  ManuallyTriggeredTask Task;
  ASSERT_SUCCESS(Task.enqueue(Queue));
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  Queue = nullptr;
  ASSERT_SUCCESS(Task.trigger());

  ASSERT_SUCCESS(olWaitEvents(Other, &Task.CompleteEvent, 1));
  ASSERT_SUCCESS(olSyncQueue(Other));
  ASSERT_SUCCESS(olSyncEvent(Task.CompleteEvent));
  ASSERT_SUCCESS(olDestroyEvent(Task.CompleteEvent));
  ASSERT_SUCCESS(olDestroyQueue(Other));
}
