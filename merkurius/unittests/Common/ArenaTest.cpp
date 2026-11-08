//===-- unittests/Common/ArenaTest.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "merkurius/Common/arena.h"
#include <optional>
#include <vector>

namespace {

using merkurius::common::Arena;

TEST(ArenaTest, SaveArrayCopies) {
  Arena arena;
  std::vector<unsigned> numbers{1, 2, 3};
  llvm::ArrayRef<unsigned> saved{
      arena.SaveArray(llvm::ArrayRef<unsigned>{numbers})};
  numbers[0] = 9;
  ASSERT_EQ(saved.size(), 3u);
  EXPECT_EQ(saved[0], 1u);
  EXPECT_EQ(saved[2], 3u);
  EXPECT_TRUE(arena.SaveArray(llvm::ArrayRef<unsigned>{}).empty());
}

// An element type that a pointer and a size convert to, so that braces
// around them would be an initializer list of two elements; so the tests
// make their arrays with parentheses too.
TEST(ArenaTest, SaveArrayOfBool) {
  Arena arena;
  const bool one[]{false};
  llvm::ArrayRef<bool> saved{arena.SaveArray(llvm::ArrayRef<bool>(one))};
  ASSERT_EQ(saved.size(), 1u);
  EXPECT_FALSE(saved[0]);
  const std::optional<bool> three[]{false, std::nullopt, true};
  llvm::ArrayRef<std::optional<bool>> optionals{
      arena.SaveArray(llvm::ArrayRef<std::optional<bool>>(three))};
  ASSERT_EQ(optionals.size(), 3u);
  EXPECT_EQ(optionals[0], false);
  EXPECT_EQ(optionals[1], std::nullopt);
  EXPECT_EQ(optionals[2], true);
}

} // namespace
