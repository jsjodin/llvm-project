//===-- unittests/Runtime/StackTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "stack.h"
#include "gtest/gtest.h"
#include <cfenv>
#include <cstdint>
#include <vector>

using namespace merkurius::runtime;

namespace {

TEST(StackTest, SizesAreRoundedUp) {
  Stack *small{NewStack(1)};
  ASSERT_NE(small, nullptr);
  EXPECT_EQ(small->bytes, std::size_t{64} << 10);
  Stack *other{NewStack((std::size_t{1} << 20) + 1)};
  ASSERT_NE(other, nullptr);
  EXPECT_EQ(other->bytes, std::size_t{2} << 20);
  EXPECT_GT(small->limit, small->base);
  EXPECT_LT(small->limit, small->base + small->bytes);
  FreeStack(small);
  FreeStack(other);
}

TEST(StackTest, ImpossibleSizeIsRefused) {
  EXPECT_EQ(NewStack(~std::size_t{0}), nullptr);
}

TEST(StackTest, FreedSlotIsReused) {
  Stack *first{NewStack(kDefaultStackBytes)};
  ASSERT_NE(first, nullptr);
  char *base{first->base};
  FreeStack(first);
  Stack *again{NewStack(kDefaultStackBytes)};
  ASSERT_NE(again, nullptr);
  EXPECT_EQ(again->base, base);
  FreeStack(again);
}

TEST(StackTest, SlotsDoNotOverlapAndCanBeTouched) {
  std::vector<Stack *> stacks;
  for (int i{0}; i < 200; ++i) {
    Stack *stack{NewStack(1 << 16)};
    ASSERT_NE(stack, nullptr);
    stack->base[0] = 1;
    stack->base[stack->bytes - 1] = 2;
    stacks.push_back(stack);
  }
  for (std::size_t i{0}; i < stacks.size(); ++i) {
    EXPECT_EQ(stacks[i]->base[0], 1);
    EXPECT_EQ(stacks[i]->base[stacks[i]->bytes - 1], 2);
    for (std::size_t j{i + 1}; j < stacks.size(); ++j) {
      EXPECT_TRUE(stacks[i]->base + stacks[i]->bytes <= stacks[j]->base ||
          stacks[j]->base + stacks[j]->bytes <= stacks[i]->base);
    }
  }
  for (Stack *stack : stacks) {
    FreeStack(stack);
  }
}

struct Coroutine {
  Context self;
  Context caller;
  long value{0};
  int rounding{FE_TONEAREST};
};

// Counts by a local on its own stack, going back to the caller each time.
void Counter(void *argument) {
  auto &coroutine{*static_cast<Coroutine *>(argument)};
  long count{100};
  for (;;) {
    coroutine.value = ++count;
    SwitchContext(coroutine.self, coroutine.caller);
  }
}

TEST(StackTest, SwitchKeepsLocalsOnBothSides) {
  Stack *stack{NewStack(kDefaultStackBytes)};
  ASSERT_NE(stack, nullptr);
  Coroutine coroutine;
  coroutine.self = NewContext(*stack, Counter, &coroutine);
  long sum{0};
  for (long i{1}; i <= 1000; ++i) {
    SwitchContext(coroutine.caller, coroutine.self);
    EXPECT_EQ(coroutine.value, 100 + i);
    sum += i;
  }
  EXPECT_EQ(sum, 500500);
  FreeStack(stack);
}

// The body starts with the stack aligned as the ABI has it.
void Aligned(void *argument) {
  auto &coroutine{*static_cast<Coroutine *>(argument)};
  coroutine.value =
      reinterpret_cast<std::uintptr_t>(__builtin_frame_address(0)) % 16;
  SwitchContext(coroutine.self, coroutine.caller);
}

TEST(StackTest, BodyIsAligned) {
  Stack *stack{NewStack(1 << 16)};
  ASSERT_NE(stack, nullptr);
  Coroutine coroutine;
  coroutine.value = -1;
  coroutine.self = NewContext(*stack, Aligned, &coroutine);
  SwitchContext(coroutine.caller, coroutine.self);
  EXPECT_EQ(coroutine.value, 0);
  FreeStack(stack);
}

// Each side keeps its own floating-point rounding mode.
void Rounding(void *argument) {
  auto &coroutine{*static_cast<Coroutine *>(argument)};
  std::fesetround(FE_UPWARD);
  for (;;) {
    SwitchContext(coroutine.self, coroutine.caller);
    coroutine.rounding = std::fegetround();
  }
}

TEST(StackTest, RoundingModeStaysWithItsSide) {
  Stack *stack{NewStack(1 << 16)};
  ASSERT_NE(stack, nullptr);
  Coroutine coroutine;
  coroutine.self = NewContext(*stack, Rounding, &coroutine);
  SwitchContext(coroutine.caller, coroutine.self);
  EXPECT_EQ(std::fegetround(), FE_TONEAREST);
  std::fesetround(FE_DOWNWARD);
  SwitchContext(coroutine.caller, coroutine.self);
  EXPECT_EQ(coroutine.rounding, FE_UPWARD);
  EXPECT_EQ(std::fegetround(), FE_DOWNWARD);
  std::fesetround(FE_TONEAREST);
  FreeStack(stack);
}

// Many computations, each on its own stack, taking turns.
TEST(StackTest, ManyTakeTurns) {
  constexpr int count{1000};
  std::vector<Coroutine> coroutines(count);
  std::vector<Stack *> stacks;
  for (Coroutine &coroutine : coroutines) {
    Stack *stack{NewStack(1 << 16)};
    ASSERT_NE(stack, nullptr);
    stacks.push_back(stack);
    coroutine.self = NewContext(*stack, Counter, &coroutine);
  }
  for (long round{1}; round <= 10; ++round) {
    for (Coroutine &coroutine : coroutines) {
      SwitchContext(coroutine.caller, coroutine.self);
      EXPECT_EQ(coroutine.value, 100 + round);
    }
  }
  for (Stack *stack : stacks) {
    FreeStack(stack);
  }
}

} // namespace
