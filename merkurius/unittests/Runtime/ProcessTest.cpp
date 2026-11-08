//===-- unittests/Runtime/ProcessTest.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "process.h"
#include "gtest/gtest.h"

using merkurius::runtime::Allocate;
using merkurius::runtime::CurrentProcess;
using merkurius::runtime::Process;
using merkurius::runtime::RunProcess;

namespace {

// Each process goes on from where it stopped, in a chunk of its own.
TEST(ProcessTest, EachHasItsOwnHeap) {
  Process a{{nullptr, nullptr}, 10};
  Process b{{nullptr, nullptr}, 11};
  RunProcess(&a);
  EXPECT_EQ(CurrentProcess(), &a);
  EXPECT_EQ(merk_heap.next, nullptr);
  char *first{static_cast<char *>(Allocate(16))};
  char *next{merk_heap.next};
  char *limit{merk_heap.limit};
  ASSERT_NE(next, nullptr);

  RunProcess(&b);
  EXPECT_EQ(CurrentProcess(), &b);
  EXPECT_EQ(merk_heap.next, nullptr);
  EXPECT_EQ(a.heap.next, next);
  EXPECT_EQ(a.heap.limit, limit);
  char *other{static_cast<char *>(Allocate(16))};
  EXPECT_FALSE(other >= first && other < limit);

  RunProcess(&a);
  EXPECT_EQ(merk_heap.next, next);
  EXPECT_EQ(merk_heap.limit, limit);
  EXPECT_EQ(Allocate(8), next);

  RunProcess(nullptr);
  EXPECT_EQ(CurrentProcess(), nullptr);
  EXPECT_EQ(merk_heap.next, nullptr);
}

} // namespace
